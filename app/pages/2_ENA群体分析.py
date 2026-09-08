import streamlit as st

from zhiji.ui import get, network_figure, setup

setup("ENA 群体过程分析")
st.info(
    "网络表示行为与情感预测状态的共现，不表示因果，也不解释某个学生的具体风险原因。"
)
stable = get("/api/v1/ena/groups/stable/network")
difficulty = get("/api/v1/ena/groups/difficulty/network")
left, right = st.columns(2)
for col, title, network in [
    (left, "稳定学习组", stable),
    (right, "持续困难组", difficulty),
]:
    with col:
        st.subheader(f"{title} · {network['unit_count']:,} 名学生")
        st.plotly_chart(network_figure(network), width="stretch")
st.caption(
    f"相同节点坐标和边宽比例 · 向前窗口 {stable['window_size_back']} · 情感阈值 {stable['affect_threshold']} · {stable['analysis_version']}"
)
st.subheader("困难组 − 稳定组")
difference = get("/api/v1/ena/difference")
st.plotly_chart(network_figure(difference), width="stretch")
st.caption("青色正边表示困难组更强，橙色负边表示稳定组更强；悬停查看具体连接与边权。")
with st.expander("边差与学生级 95% 置信区间（探索性，未作多重比较校正）"):
    st.dataframe(difference["edges"], hide_index=True)
