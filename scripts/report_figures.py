import json
from pathlib import Path

import plotly.graph_objects as go


def main():
    metrics = json.loads(Path("reports/evaluation/test_metrics.json").read_text())
    registry = json.loads(Path("artifacts/registry.json").read_text())
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="理想校准",
            line={"dash": "dash", "color": "#999"},
        )
    )
    for key, label in [
        ("current_prediction_model", "下一题上下文未知"),
        ("current_prediction_model_known", "下一题上下文已知"),
    ]:
        curve = metrics[registry[key]]["metrics"]["calibration_curve"]
        figure.add_trace(
            go.Scatter(
                x=curve["predicted"],
                y=curve["observed"],
                mode="lines+markers",
                name=label,
            )
        )
    figure.update_layout(
        title="测试集风险概率校准",
        xaxis_title="模型预测答错概率",
        yaxis_title="实际答错比例",
        template="plotly_white",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1], "scaleanchor": "x"},
    )
    out = Path("reports/figures/calibration.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(out, include_plotlyjs=True)
    print(out)


if __name__ == "__main__":
    main()
