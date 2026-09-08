import os

import httpx
import plotly.graph_objects as go
import streamlit as st

LABELS = {
    "incorrect": "答错",
    "hint_used": "使用提示",
    "bottom_hint_used": "最终提示",
    "repeated_attempt": "重复尝试",
    "slow_response": "慢反应",
    "frustrated": "挫败预测",
    "confused": "困惑预测",
    "bored": "无聊预测",
    "concentrating": "专注预测",
}
RISK = {"high": "▲ 高风险", "medium": "◆ 中风险", "low": "● 低风险"}


def setup(title):
    st.set_page_config(page_title=title + " · 知迹", page_icon="◈", layout="wide")
    st.caption("知迹 ZHIJI  /  学习过程观察与教师复核")
    st.title(title)


def get(path):
    try:
        response = httpx.get(
            os.environ.get("ZHIJI_API_URL", "http://127.0.0.1:8000") + path, timeout=20
        )
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        st.error("暂时无法读取结果。请确认本地 API 已启动，且模型与分析产物已生成。")
        st.stop()


def network_figure(network):
    nodes = {n["id"]: n for n in network["nodes"]}
    figure = go.Figure()
    for edge in network["edges"]:
        a, b = nodes[edge["source"]], nodes[edge["target"]]
        weight = edge["weight"]
        figure.add_trace(
            go.Scatter(
                x=[a["x"], b["x"]],
                y=[a["y"], b["y"]],
                mode="lines",
                line={
                    "width": max(0.2, abs(weight) * 15),
                    "color": "#C57547" if weight < 0 else "#288D91",
                },
                text=f"{LABELS[a['id']]} — {LABELS[b['id']]}：{weight:.4f}",
                hoverinfo="text",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[n["x"] for n in nodes.values()],
            y=[n["y"] for n in nodes.values()],
            mode="markers+text",
            text=[LABELS[n["id"]] for n in nodes.values()],
            textposition="top center",
            marker={"size": 14, "color": "#153A49"},
            showlegend=False,
        )
    )
    figure.update_layout(
        height=470,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
    )
    return figure
