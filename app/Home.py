import pandas as pd
import plotly.express as px
import streamlit as st

from zhiji.ui import RISK, get, setup

setup("学情概览")
st.info("公开历史数据的匿名演示样本。预测供教师复核，不替代教学判断。")
model = get("/api/v1/models/current")
response = get("/api/v1/students")
items = response["items"]
if not items:
    st.warning("尚未生成演示学生，请先运行演示数据导出命令。")
    st.stop()
frame = pd.DataFrame(items)
counts = frame.risk_level.value_counts()
cols = st.columns(4)
for col, label, value in zip(
    cols,
    ["演示学生", "▲ 高风险", "◆ 中风险", "● 低风险"],
    [len(items), counts.get("high", 0), counts.get("medium", 0), counts.get("low", 0)],
):
    col.metric(label, int(value))
st.caption(
    f"模型 {model['model_version']} · 风险阈值：中 {model['risk_thresholds']['medium']:.1%} / 高 {model['risk_thresholds']['high']:.1%} · 阈值在验证集按答错召回目标选择"
)
left, right = st.columns([2, 1])
with left:
    st.subheader("待复核学生")
    view = frame[
        ["student_id", "risk_probability", "risk_level", "generated_at"]
    ].copy()
    view["risk_level"] = view.risk_level.map(RISK)
    view = view.rename(
        columns={
            "student_id": "匿名编号",
            "risk_probability": "风险概率",
            "risk_level": "风险等级",
            "generated_at": "结果生成时间",
        }
    )
    st.dataframe(
        view.sort_values("风险概率", ascending=False),
        hide_index=True,
        width="stretch",
        column_config={
            "风险概率": st.column_config.ProgressColumn(
                format="%.2f", min_value=0, max_value=1
            )
        },
    )
    st.page_link("pages/1_学生详情.py", label="查看学生轨迹与近期证据 →")
with right:
    st.subheader("风险概率分布")
    st.plotly_chart(
        px.histogram(
            frame,
            x="risk_probability",
            nbins=10,
            color_discrete_sequence=["#288D91"],
            labels={"risk_probability": "风险概率"},
        ),
        width="stretch",
    )
    st.caption("分布描述当前演示样本，不代表真实班级的风险分布。")
