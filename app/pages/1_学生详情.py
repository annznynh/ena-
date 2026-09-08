import pandas as pd
import plotly.express as px
import streamlit as st

from zhiji.ui import RISK, get, setup

setup("学生轨迹与证据")
students = get("/api/v1/students")["items"]
if not students:
    st.warning("暂无演示学生")
    st.stop()
sid = st.selectbox("匿名学生", [x["student_id"] for x in students])
current = next(x for x in students if x["student_id"] == sid)
cols = st.columns(3)
cols[0].metric("下一次答错概率", f"{current['risk_probability']:.1%}")
cols[1].metric("风险等级", RISK[current["risk_level"]])
cols[2].metric("预测正确概率", f"{current['correct_probability']:.1%}")
st.caption(current["warning"])
timeline = get(f"/api/v1/students/{sid}/risk-timeline")["items"]
st.subheader("历史锚点风险变化")
st.plotly_chart(
    px.line(
        pd.DataFrame(timeline),
        x="occurred_at",
        y="risk_probability",
        markers=True,
        labels={"occurred_at": "原始日志时间", "risk_probability": "下一次答错概率"},
        range_y=[0, 1],
    ),
    width="stretch",
)
st.caption("曲线使用各锚点以前的历史。原始日志没有时区信息，时间按原始记录展示。")
recent = get(f"/api/v1/students/{sid}/recent-evidence")
left, right = st.columns(2)
for col, kind, title in [
    (left, "recent_fact", "近期事实"),
    (right, "model_contribution", "模型贡献"),
]:
    with col:
        st.subheader(title)
        for item in recent["evidence"]:
            if item["kind"] == kind:
                st.write("• " + item["value"])
st.subheader("最近 20 次交互")
history = pd.DataFrame(recent["history"])
st.dataframe(
    history[
        ["occurred_at", "correct", "hint_count", "attempt_count", "ms_first_response"]
    ].rename(
        columns={
            "occurred_at": "日志时间",
            "correct": "是否正确",
            "hint_count": "提示次数",
            "attempt_count": "尝试次数",
            "ms_first_response": "首次反应毫秒",
        }
    ),
    hide_index=True,
    width="stretch",
)
with st.expander("情感字段：系统预测置信度"):
    st.dataframe(
        pd.DataFrame(history.affect.tolist()).rename(
            columns={
                "frustrated": "挫败预测",
                "confused": "困惑预测",
                "concentrating": "专注预测",
                "bored": "无聊预测",
            }
        ),
        hide_index=True,
    )
st.page_link("pages/2_ENA群体分析.py", label="了解群体过程网络（不是个体诊断） →")
