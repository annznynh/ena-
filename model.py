import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 配置参数
# ==============================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
DATA_PATH     = os.path.join(BASE_DIR, "2012-2013-data-with-predictions-4-final.csv")
RESULT_DIR    = os.path.join(BASE_DIR, "result")
EDA_DIR       = os.path.join(RESULT_DIR, "eda")   # ★ 预处理图表专属目录

N_ROWS        = 300000
TEST_SIZE     = 0.2
VAL_SIZE      = 0.2
RANDOM_STATE  = 42
BATCH_SIZE    = 64
EPOCHS        = 50
PATIENCE      = 5
LEARNING_RATE = 1e-3
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建所有目录
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(EDA_DIR,    exist_ok=True)

print(f"脚本目录 : {BASE_DIR}")
print(f"结果目录 : {RESULT_DIR}")
print(f"EDA目录  : {EDA_DIR}")
print(f"使用设备 : {DEVICE}")

# ==============================
# 1. 加载数据
# ==============================
print("\nLoading data...")
df = pd.read_csv(DATA_PATH, nrows=N_ROWS, encoding='utf-8')
print(f"Data shape: {df.shape}")

# ==============================
# 2. 数据预处理 + 可视化（共14张图）
# ==============================
print("\nPreprocessing + Visualization...")

# --------------------------------------------------
# 【图01】原始数据缺失值分析
# --------------------------------------------------
missing_counts = df.isnull().sum()
missing_ratio  = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'column':        missing_counts.index,
    'missing_count': missing_counts.values,
    'missing_ratio': missing_ratio.values
}).query('missing_count > 0').sort_values('missing_ratio', ascending=False)

if not missing_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(missing_df) * 0.5)))

    axes[0].barh(missing_df['column'], missing_df['missing_count'],
                 color='steelblue', edgecolor='white')
    axes[0].set_xlabel('Missing Count')
    axes[0].set_title('Missing Values Count per Column', fontweight='bold')
    axes[0].invert_yaxis()
    for i, v in enumerate(missing_df['missing_count']):
        axes[0].text(v, i, f'  {v:,}', va='center', fontsize=8)

    bar_colors = ['#98C9F1' if r > 50 else "#B4F3B4" if r > 20 else "#fffb7a"
                  for r in missing_df['missing_ratio']]
    axes[1].barh(missing_df['column'], missing_df['missing_ratio'],
                 color=bar_colors, edgecolor='white')
    axes[1].set_xlabel('Missing Ratio (%)')
    axes[1].set_title('Missing Values Ratio per Column', fontweight='bold')
    axes[1].axvline(x=50, color='red',    linestyle='--', alpha=0.7, label='>50%')
    axes[1].axvline(x=20, color='orange', linestyle='--', alpha=0.7, label='>20%')
    axes[1].invert_yaxis()
    axes[1].legend()
    for i, v in enumerate(missing_df['missing_ratio']):
        axes[1].text(v, i, f'  {v:.1f}%', va='center', fontsize=8)

    plt.suptitle('Raw Data: Missing Values Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '01_missing_values.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_missing_values.png")
else:
    print("  无缺失值，跳过图01")

# --------------------------------------------------
# 【图02】目标变量分布（答对/答错比例）
# --------------------------------------------------
if 'correct' in df.columns:
    # ★ 修复：先打印实际类别，动态生成labels，不写死
    print(f"  correct列的唯一值: {df['correct'].unique()}")
    print(f"  correct列的值计数:\n{df['correct'].value_counts()}")

    correct_counts = df['correct'].value_counts().sort_index()
    total_s        = correct_counts.sum()

    # ★ 动态生成标签（根据实际有几个类别）
    actual_labels = [str(idx) for idx in correct_counts.index]
    pie_colors    = ["#FF7B78", "#92CEFF", "#A2FFA6", "#FFCD81"][:len(correct_counts)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 饼图
    _, _, autotexts = axes[0].pie(
        correct_counts.values,
        labels=actual_labels,          # ★ 用动态生成的labels
        colors=pie_colors,
        autopct='%1.2f%%',
        startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight('bold')
    axes[0].set_title('Target Distribution (Pie)', fontweight='bold')

    # 柱状图
    bars = axes[1].bar(
        actual_labels,                 # ★ 用动态生成的labels
        correct_counts.values,
        color=pie_colors,
        edgecolor='white',
        width=0.5
    )
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel('correct')
    axes[1].set_title('Target Distribution (Bar)', fontweight='bold')
    axes[1].set_ylim(0, max(correct_counts.values) * 1.2)
    for bar, cnt in zip(bars, correct_counts.values):
        axes[1].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f'{cnt:,}\n({cnt/total_s*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold'
        )

    # 如果恰好有两类，显示不平衡比例
    if len(correct_counts) == 2:
        ratio = correct_counts.max() / max(correct_counts.min(), 1)
        axes[1].text(
            0.97, 0.95,
            f'Imbalance ≈ {ratio:.1f}:1',
            transform=axes[1].transAxes,
            ha='right', va='top', color='red', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )

    plt.suptitle('Target Variable (correct) Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '02_target_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_target_distribution.png")


# --------------------------------------------------
# 时间字段解析
# --------------------------------------------------
def parse_end_time(t):
    if pd.isna(t):
        return np.nan
    t_str = str(t).strip()
    if ('-' in t_str or '/' in t_str) and ':' in t_str:
        return np.nan
    t_str = t_str.replace(',', '.')
    parts = t_str.split(':')
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            return np.nan
    except:
        return np.nan

df['duration_seconds'] = df['end_time'].apply(parse_end_time)
df['start_time']       = pd.to_datetime(df['start_time'], format='%Y/%m/%d %H:%M', errors='coerce')
df['start_hour']       = df['start_time'].dt.hour
df['start_dayofweek']  = df['start_time'].dt.dayofweek

# --------------------------------------------------
# 【图03】时间特征分布
# --------------------------------------------------
if df['start_hour'].notna().sum() > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    hour_counts = df['start_hour'].value_counts().sort_index()
    hours_full  = pd.Series(0, index=range(24))
    hours_full.update(hour_counts)
    h_colors = ["#FECC80" if 6 <= h < 12 else
                "#FF8B89" if 12 <= h < 18 else
                "#9CA8E7" if 18 <= h < 24 else '#90A4AE'
                for h in range(24)]
    axes[0].bar(hours_full.index, hours_full.values, color=h_colors, edgecolor='white')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Answer Activity by Hour of Day', fontweight='bold')
    axes[0].set_xticks(range(24))

    day_names  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_counts = df['start_dayofweek'].value_counts().sort_index()
    days_full  = pd.Series(0, index=range(7))
    days_full.update(day_counts)
    d_colors = ["#FF8280" if i >= 5 else "#75BCF7" for i in range(7)]
    axes[1].bar(range(7), days_full.values, color=d_colors, edgecolor='white')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Answer Activity by Day of Week', fontweight='bold')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_names)
    axes[1].axvspan(4.5, 6.5, alpha=0.1, color='red', label='Weekend')
    axes[1].legend()

    plt.suptitle('Time Feature Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '03_time_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_time_features.png")

df.drop(['start_time', 'end_time'], axis=1, inplace=True)
df.drop(['actions', 'answer_text', 'first_action'], axis=1, inplace=True, errors='ignore')

numeric_cols = ['ms_first_response', 'hint_count', 'attempt_count', 'overlap_time']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].replace('#NAME?', np.nan)

# --------------------------------------------------
# 【图04】关键数值特征分布 + 箱线图（清洗前）
# --------------------------------------------------
key_num_cols = [c for c in numeric_cols + ['duration_seconds'] if c in df.columns]
if key_num_cols:
    n_c = len(key_num_cols)
    fig, axes = plt.subplots(2, n_c, figsize=(5 * n_c, 10))
    if n_c == 1:
        axes = axes.reshape(2, 1)

    for i, col in enumerate(key_num_cols):
        data = df[col].dropna()
        if len(data) == 0:
            continue
        clip_data = data.clip(data.quantile(0.01), data.quantile(0.99))

        axes[0, i].hist(clip_data, bins=50, color='#42A5F5', edgecolor='white', alpha=0.8)
        axes[0, i].axvline(data.mean(),   color='red',   linestyle='--',
                           linewidth=1.5, label=f'Mean={data.mean():.1f}')
        axes[0, i].axvline(data.median(), color='green', linestyle='-.',
                           linewidth=1.5, label=f'Median={data.median():.1f}')
        axes[0, i].set_title(f'{col}\n(Before Cleaning)', fontweight='bold')
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('Count')
        axes[0, i].legend(fontsize=8)

        bp = axes[1, i].boxplot(
            data.clip(data.quantile(0.001), data.quantile(0.999)),
            patch_artist=True,
            flierprops=dict(marker='o', markersize=2, markerfacecolor='red', alpha=0.3)
        )
        bp['boxes'][0].set_facecolor('#90CAF9')
        bp['medians'][0].set_color('#E53935')
        bp['medians'][0].set_linewidth(2)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        n_out = ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum()
        axes[1, i].set_title(f'{col} Boxplot', fontweight='bold')
        axes[1, i].text(1.08, 0.5,
                        f"N={len(data):,}\nMean={data.mean():.1f}\n"
                        f"Std={data.std():.1f}\nOutliers\n={n_out:,}\n"
                        f"({n_out/len(data)*100:.1f}%)",
                        transform=axes[1, i].transAxes, va='center', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Numeric Features: Distribution & Outlier Detection (Before Cleaning)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '04_numeric_distribution_before.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_numeric_distribution_before.png")

# --------------------------------------------------
# 填充缺失值
# --------------------------------------------------
print("Filling missing values...")
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0
        df[col] = df[col].fillna(median_val)
    else:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])
        else:
            df[col] = df[col].fillna('missing')

# --------------------------------------------------
# 【图05】填充后分布
# --------------------------------------------------
if key_num_cols:
    n_c = len(key_num_cols)
    fig, axes = plt.subplots(1, n_c, figsize=(5 * n_c, 5))
    if n_c == 1:
        axes = [axes]
    for i, col in enumerate(key_num_cols):
        data = df[col]
        axes[i].hist(data, bins=50, color='#66BB6A', edgecolor='white', alpha=0.8)
        axes[i].axvline(data.mean(),   color='red',  linestyle='--', linewidth=2,
                        label=f'Mean={data.mean():.2f}')
        axes[i].axvline(data.median(), color='blue', linestyle='-.', linewidth=2,
                        label=f'Median={data.median():.2f}')
        axes[i].set_title(f'{col} (After Fill)', fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].legend(fontsize=9)
    plt.suptitle('Numeric Features After Missing Value Filling', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '05_numeric_after_fill.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_numeric_after_fill.png")

# --------------------------------------------------
# 去除异常值（IQR）
# --------------------------------------------------
print("Removing outliers...")
df_before = df.copy()
outlier_mask = pd.Series(False, index=df.index)
outlier_info = {}
for col in numeric_cols + ['duration_seconds']:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        col_out = (df[col] < lower) | (df[col] > upper)
        outlier_mask = outlier_mask | col_out
        outlier_info[col] = {
            'lower': lower, 'upper': upper,
            'n_outliers': int(col_out.sum()),
            'ratio': float(col_out.mean() * 100)
        }
df = df[~outlier_mask].reset_index(drop=True)
print(f"After outlier removal: {df.shape}")

# --------------------------------------------------
# 【图06】异常值去除前后对比
# --------------------------------------------------
if key_num_cols:
    n_c = len(key_num_cols)
    fig, axes = plt.subplots(2, n_c, figsize=(5 * n_c, 10))
    if n_c == 1:
        axes = axes.reshape(2, 1)
    for i, col in enumerate(key_num_cols):
        info = outlier_info.get(col, {})
        axes[0, i].hist(df_before[col], bins=60, color='#EF5350', edgecolor='white', alpha=0.7)
        if info:
            axes[0, i].axvline(info['lower'], color='purple', linestyle='--',
                               linewidth=2, label=f'Lower={info["lower"]:.1f}')
            axes[0, i].axvline(info['upper'], color='orange', linestyle='--',
                               linewidth=2, label=f'Upper={info["upper"]:.1f}')
        axes[0, i].set_title(
            f'{col}  Before\nOutliers:{info.get("n_outliers",0):,} ({info.get("ratio",0):.1f}%)',
            fontweight='bold', fontsize=10)
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('Count')
        axes[0, i].legend(fontsize=8)

        axes[1, i].hist(df[col], bins=60, color='#42A5F5', edgecolor='white', alpha=0.7)
        axes[1, i].set_title(f'{col}  After\nRemaining:{len(df):,}',
                             fontweight='bold', fontsize=10)
        axes[1, i].set_xlabel(col)
        axes[1, i].set_ylabel('Count')

    n_removed = len(df_before) - len(df)
    plt.suptitle(
        f'Outlier Removal Before vs After (IQR)\n'
        f'Removed {n_removed:,} rows ({n_removed/len(df_before)*100:.1f}%)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '06_outlier_removal_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_outlier_removal_comparison.png")

# --------------------------------------------------
# 【图07】各列异常值汇总
# --------------------------------------------------
if outlier_info:
    cols_oi = list(outlier_info.keys())
    n_outs  = [outlier_info[c]['n_outliers'] for c in cols_oi]
    ratios  = [outlier_info[c]['ratio']      for c in cols_oi]
    bc      = ['#EF5350' if r > 10 else '#FFA726' if r > 5 else '#66BB6A' for r in ratios]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(cols_oi, n_outs, color=bc, edgecolor='white')
    axes[0].set_xlabel('Feature')
    axes[0].set_ylabel('Outlier Count')
    axes[0].set_title('Outlier Count by Feature', fontweight='bold')
    axes[0].set_xticklabels(cols_oi, rotation=20, ha='right')
    for bar, n in zip(axes[0].patches, n_outs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{n:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[1].bar(cols_oi, ratios, color=bc, edgecolor='white')
    axes[1].set_xlabel('Feature')
    axes[1].set_ylabel('Outlier Ratio (%)')
    axes[1].set_title('Outlier Ratio by Feature', fontweight='bold')
    axes[1].set_xticklabels(cols_oi, rotation=20, ha='right')
    axes[1].axhline(y=5,  color='orange', linestyle='--', alpha=0.7, label='5%')
    axes[1].axhline(y=10, color='red',    linestyle='--', alpha=0.7, label='10%')
    axes[1].legend()
    for bar, r in zip(axes[1].patches, ratios):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{r:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Outlier Summary by Feature', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '07_outlier_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_outlier_summary.png")

# --------------------------------------------------
# 类别编码
# --------------------------------------------------
print("Encoding categorical variables...")
label_encoders   = {}
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'correct' in categorical_cols:
    categorical_cols.remove('correct')
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --------------------------------------------------
# 【图08】特征与目标变量关系（分组箱线图）
# --------------------------------------------------
if 'correct' in df.columns:
    plot_cols = [c for c in numeric_cols + ['duration_seconds'] if c in df.columns]
    if plot_cols:
        n_c = len(plot_cols)
        fig, axes = plt.subplots(1, n_c, figsize=(5 * n_c, 6))
        if n_c == 1:
            axes = [axes]
        for i, col in enumerate(plot_cols):
            d0 = df[df['correct'] == 0][col]
            d1 = df[df['correct'] == 1][col]
            bp = axes[i].boxplot([d0, d1], labels=['Incorrect', 'Correct'],
                                 patch_artist=True,
                                 flierprops=dict(marker='o', markersize=2, alpha=0.3))
            bp['boxes'][0].set_facecolor('#EF9A9A')
            bp['boxes'][1].set_facecolor('#90CAF9')
            for med in bp['medians']:
                med.set_color('black')
                med.set_linewidth(2)
            axes[i].set_title(f'{col}\nvs Target', fontweight='bold')
            axes[i].set_ylabel(col)
            axes[i].text(1, d0.median(), f'  {d0.median():.1f}', va='center',
                         fontsize=9, color='#C62828')
            axes[i].text(2, d1.median(), f'  {d1.median():.1f}', va='center',
                         fontsize=9, color='#1565C0')
        plt.suptitle('Feature Distribution: Correct vs Incorrect', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, '08_feature_vs_target.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 08_feature_vs_target.png")

# --------------------------------------------------
# 【图09】数值特征相关性热力图
# --------------------------------------------------
num_df = df.select_dtypes(include=['int64', 'float64'])
if len(num_df.columns) >= 2:
    corr_matrix = num_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sz = max(10, len(corr_matrix.columns))
    fig, ax = plt.subplots(figsize=(sz, int(sz * 0.8)))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, square=True, linewidths=0.5,
                annot_kws={'size': 7}, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '09_correlation_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 09_correlation_heatmap.png")

# ==============================
# 3. 特征目标分离，移除ID列
# ==============================
X = df.drop('correct', axis=1)
y = df['correct'].astype(int)

id_cols = ['user_id', 'problem_id', 'problem_log_id', 'problemlogid',
           'assignment_id', 'assistment_id', 'teacher_id', 'school_id',
           'template_id', 'base_sequence_id', 'sequence_id']
id_cols = [col for col in id_cols if col in X.columns]
X = X.drop(columns=id_cols)
print(f"Removed ID columns: {id_cols}")

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VAL_SIZE,
    random_state=RANDOM_STATE, stratify=y_train_val)
print(f"Train:{X_train.shape[0]} | Val:{X_val.shape[0]} | Test:{X_test.shape[0]}")

# --------------------------------------------------
# 【图10】数据集划分比例
# --------------------------------------------------
n_train = X_train.shape[0]
n_val   = X_val.shape[0]
n_test  = X_test.shape[0]
total   = n_train + n_val + n_test
s_colors = ['#42A5F5', '#FFA726', '#EF5350']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels_s = [f'Train\n{n_train:,}\n({n_train/total*100:.1f}%)',
            f'Val\n{n_val:,}\n({n_val/total*100:.1f}%)',
            f'Test\n{n_test:,}\n({n_test/total*100:.1f}%)']
axes[0].pie([n_train, n_val, n_test], labels=labels_s, colors=s_colors,
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
axes[0].set_title('Dataset Split (Pie)', fontweight='bold')

axes[1].barh(['Dataset'], [n_train], color=s_colors[0], label=f'Train ({n_train:,})')
axes[1].barh(['Dataset'], [n_val],   left=[n_train],         color=s_colors[1], label=f'Val ({n_val:,})')
axes[1].barh(['Dataset'], [n_test],  left=[n_train + n_val], color=s_colors[2], label=f'Test ({n_test:,})')
axes[1].set_xlabel('Number of Samples')
axes[1].set_title('Dataset Split (Stacked Bar)', fontweight='bold')
axes[1].legend(fontsize=10)
for val, left in [(n_train, 0), (n_val, n_train), (n_test, n_train+n_val)]:
    if val > total * 0.05:
        axes[1].text(left + val/2, 0, f'{val:,}',
                     ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.suptitle(f'Dataset Split Overview  (Total: {total:,})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '10_dataset_split.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 10_dataset_split.png")

# --------------------------------------------------
# 【图11】三个集合的目标变量分布验证
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (sname, y_s) in zip(axes, [('Train', y_train), ('Val', y_val), ('Test', y_test)]):
    cnts  = y_s.value_counts().sort_index()
    bars  = ax.bar(['Incorrect', 'Correct'], cnts.values,
                   color=['#EF5350', '#42A5F5'], edgecolor='white', width=0.5)
    ax.set_title(f'{sname} Set  (n={len(y_s):,})', fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_ylim(0, max(cnts.values) * 1.2)
    for bar, cnt in zip(bars, cnts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{cnt:,}\n({cnt/len(y_s)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
plt.suptitle('Target Distribution: Train / Val / Test (Stratified Sampling Verification)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '11_split_target_distribution.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 11_split_target_distribution.png")

# ==============================
# 4. 随机森林特征重要性
# ==============================
print("\nTraining Random Forest for feature importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)

feat_imp_df = pd.DataFrame({
    'feature':    X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# 原有图存RESULT_DIR
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feat_imp_df.head(20))
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'feature_importance.png'))
plt.close()

# --------------------------------------------------
# 【图12】特征重要性详细图 + 累积曲线
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(feat_imp_df) * 0.3)))
imp_colors = ['#1565C0' if i < 10 else '#42A5F5' if i < 20 else '#90CAF9'
              for i in range(len(feat_imp_df))]
axes[0].barh(feat_imp_df['feature'], feat_imp_df['importance'],
             color=imp_colors, edgecolor='white')
axes[0].invert_yaxis()
axes[0].set_xlabel('Importance Score')
axes[0].set_title('All Features Importance\n(Dark=Top10, Mid=Top20, Light=Rest)',
                  fontweight='bold')
axes[0].axvline(feat_imp_df['importance'].mean(), color='red', linestyle='--',
                alpha=0.7, label=f'Mean={feat_imp_df["importance"].mean():.4f}')
axes[0].legend()

cumsum = feat_imp_df['importance'].cumsum().values
axes[1].plot(range(1, len(cumsum)+1), cumsum, 'b-o', markersize=4, linewidth=2)
for thresh, color, lbl in [(0.80, 'orange', '80%'), (0.90, 'red', '90%'), (0.95, 'purple', '95%')]:
    axes[1].axhline(y=thresh, color=color, linestyle='--', linewidth=1.5, label=lbl)
    n_need = int(np.argmax(cumsum >= thresh)) + 1
    axes[1].axvline(x=n_need, color=color, linestyle=':', alpha=0.5)
    axes[1].text(n_need + 0.3, thresh - 0.03, f'Top{n_need}',
                 color=color, fontsize=9, fontweight='bold')
axes[1].set_xlabel('Number of Features(个)')
axes[1].set_ylabel('Cumulative Importance(累计比例)')
axes[1].set_title('Cumulative Feature Importance', fontweight='bold')
axes[1].legend()
axes[1].set_ylim(0, 1.05)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Random Forest Feature Importance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '12_feature_importance_detail.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 12_feature_importance_detail.png")

top_features = feat_imp_df.head(10)['feature'].tolist()
print(f"Top 10 features: {top_features}")
X_train = X_train[top_features]
X_val   = X_val[top_features]
X_test  = X_test[top_features]

# ==============================
# 5. 数据标准化
# ==============================
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# --------------------------------------------------
# 【图13】标准化前后对比
# --------------------------------------------------
n_show = min(6, len(top_features))
fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
if n_show == 1:
    axes = axes.reshape(2, 1)
for i, col in enumerate(top_features[:n_show]):
    raw    = X_train[col].values
    scaled = X_train_scaled[:, i]
    axes[0, i].hist(raw, bins=40, color='#EF5350', edgecolor='white', alpha=0.8)
    axes[0, i].set_title(f'{col}\nBefore Scaling', fontweight='bold', fontsize=9)
    axes[0, i].set_xlabel(f'Mean={np.mean(raw):.2f}  Std={np.std(raw):.2f}', fontsize=8)
    axes[0, i].set_ylabel('Count')
    axes[1, i].hist(scaled, bins=40, color='#42A5F5', edgecolor='white', alpha=0.8)
    axes[1, i].set_title(f'{col}\nAfter Scaling', fontweight='bold', fontsize=9)
    axes[1, i].set_xlabel(f'Mean={np.mean(scaled):.4f}  Std={np.std(scaled):.4f}', fontsize=8)
    axes[1, i].set_ylabel('Count')
    axes[1, i].axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

plt.suptitle('Feature Scaling: Before vs After StandardScaler (Mean≈0, Std≈1)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '13_scaling_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 13_scaling_comparison.png")

# --------------------------------------------------
# 【图14】预处理流程总览
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

steps = ['Raw\nData', 'Drop\nCols', 'Fill\nNA', 'Remove\nOutliers', 'Encode', 'Feature\nSelect']
sizes = [N_ROWS, N_ROWS, N_ROWS, len(df), len(df), len(df)]
axes[0].plot(range(len(steps)), sizes, 'b-o', markersize=10, linewidth=2.5)
axes[0].fill_between(range(len(steps)), sizes, alpha=0.1, color='blue')
axes[0].set_xticks(range(len(steps)))
axes[0].set_xticklabels(steps, fontsize=10)
axes[0].set_ylabel('Number of Samples')
axes[0].set_title('Sample Size Through Preprocessing', fontweight='bold')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
for i, sz in enumerate(sizes):
    axes[0].text(i, sz, f'\n{sz:,}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#1565C0')

axes[1].axis('off')
n_removed = N_ROWS - len(df)
info_lines = [
    ("Step 1  Load Data",         f"{N_ROWS:,} rows from CSV"),
    ("Step 2  Parse Time",        "end_time→seconds, extract hour/weekday"),
    ("Step 3  Drop Columns",      "Remove actions/answer_text/first_action"),
    ("Step 4  Fill Missing",      "Numeric:median  Categorical:mode"),
    ("Step 5  Remove Outliers",   f"IQR → removed {n_removed:,} rows ({n_removed/N_ROWS*100:.1f}%)"),
    ("Step 6  Label Encoding",    "Categorical → Integer"),
    ("Step 7  Feature Selection", f"Random Forest Top10"),
    ("Step 8  Standardization",   "StandardScaler: Mean=0, Std=1"),
    ("Final   Dataset",           f"Train:{n_train:,}  Val:{n_val:,}  Test:{n_test:,}"),
]
y_pos = 0.95
for title, detail in info_lines:
    axes[1].text(0.02, y_pos, f"▶ {title}", fontsize=10, fontweight='bold',
                 color='#1565C0', transform=axes[1].transAxes)
    axes[1].text(0.05, y_pos - 0.044, detail, fontsize=9,
                 color='#37474F', transform=axes[1].transAxes)
    y_pos -= 0.10
axes[1].set_title('Preprocessing Pipeline Summary', fontweight='bold')
axes[1].add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
                                 edgecolor='lightgray', linewidth=1,
                                 transform=axes[1].transAxes))

plt.suptitle('Data Preprocessing Overview', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '14_preprocessing_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 14_preprocessing_summary.png")

print(f"\n{'='*50}")
print(f"✅ 预处理可视化完成！14张图 → {EDA_DIR}")
print(f"{'='*50}\n")

# ==============================
# 5（续）转为PyTorch张量
# ==============================
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_val_tensor   = torch.tensor(X_val_scaled,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val.values,   dtype=torch.long)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test.values,  dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)
test_dataset  = TensorDataset(X_test_tensor,  y_test_tensor)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# 6. 模型定义（与原始完全一致）
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm    = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc      = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, lstm_output):
        w = torch.tanh(self.attention(lstm_output))
        w = torch.softmax(w, dim=1)
        return torch.sum(w * lstm_output, dim=1)

class AttentionLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.3):
        super(AttentionLSTMModel, self).__init__()
        self.lstm      = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = AttentionLayer(hidden_dim)
        self.fc        = nn.Linear(hidden_dim, output_dim)
        self.dropout   = nn.Dropout(dropout)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        return self.fc(self.dropout(context))

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=1, output_dim=2, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc      = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        return self.fc(self.dropout(x))

# ==============================
# 7. 训练函数
# ==============================
def train_model(model, train_loader, val_loader, epochs, patience, lr, device, model_name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc, best_model_state, patience_counter = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            _, pred = torch.max(outputs, 1)
            total   += y_batch.size(0)
            correct += (pred == y_batch).sum().item()

        epoch_train_loss = total_loss / total
        epoch_train_acc  = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs  = model(X_batch)
                loss     = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, pred  = torch.max(outputs, 1)
                total    += y_batch.size(0)
                correct  += (pred == y_batch).sum().item()

        epoch_val_loss = val_loss / total
        epoch_val_acc  = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        scheduler.step(epoch_val_acc)
        print(f"{model_name} Epoch {epoch+1}/{epochs}: "
              f"Train Loss:{epoch_train_loss:.4f} Acc:{epoch_train_acc:.4f} | "
              f"Val Loss:{epoch_val_loss:.4f} Acc:{epoch_val_acc:.4f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc     = epoch_val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accs, val_accs

# ==============================
# 8. 评估函数
# ==============================
def evaluate_model(model, test_loader, device, model_name, result_dir,
                   train_losses, val_losses, train_accs, val_accs):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs   = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred,    zero_division=0)
    f1        = f1_score(y_true, y_pred,         zero_division=0)
    report    = classification_report(y_true, y_pred,
                                      target_names=['Incorrect', 'Correct'], zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Incorrect', 'Correct'],
                yticklabels=['Incorrect', 'Correct'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

    fpr, tpr, _   = roc_curve(y_true, y_prob)
    roc_auc_value = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_value:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{model_name}_roc.png'))
    plt.close()

    with open(os.path.join(result_dir, f'{model_name}_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"AUC:       {roc_auc_value:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses,   label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs,   label='Val Acc')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{model_name}_training_curves.png'))
    plt.close()

    return accuracy, precision, recall, f1, roc_auc_value

# ==============================
# 9. 训练和评估三个模型
# ==============================
models_config = {
    'LSTM':          LSTMModel(input_dim=len(top_features)),
    'AttentionLSTM': AttentionLSTMModel(input_dim=len(top_features)),
    'Transformer':   TransformerModel(input_dim=len(top_features))
}
all_metrics = {}
for name, model in models_config.items():
    print(f"\nTraining {name}...")
    model_dir = os.path.join(RESULT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    trained_model, train_loss, val_loss, train_acc, val_acc = train_model(
        model, train_loader, val_loader, EPOCHS, PATIENCE, LEARNING_RATE, DEVICE, name)
    acc, prec, rec, f1, roc_auc_val = evaluate_model(
        trained_model, test_loader, DEVICE, name, model_dir,
        train_loss, val_loss, train_acc, val_acc)
    all_metrics[name] = {'Accuracy': acc, 'Precision': prec,
                         'Recall': rec, 'F1 Score': f1, 'AUC': roc_auc_val}

# ==============================
# 10. 模型对比
# ==============================
metrics_df = pd.DataFrame(all_metrics).T
metrics_df.to_csv(os.path.join(RESULT_DIR, 'model_comparison.csv'))
plt.figure(figsize=(12, 6))
metrics_melted = metrics_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
sns.barplot(x='Metric', y='Score', hue='index', data=metrics_melted)
plt.title('Model Comparison')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'model_comparison.png'))
plt.close()

print("\nAll done! Results saved in 'result' folder.")
