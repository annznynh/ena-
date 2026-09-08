import json
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, FiniteFloat

from zhiji.ena.codes import CODES


class Node(BaseModel):
    id: str
    label: str
    x: FiniteFloat
    y: FiniteFloat


class Edge(BaseModel):
    source: str
    target: str
    weight: FiniteFloat


class Network(BaseModel):
    nodes: list[Node] = Field(min_length=9, max_length=9)
    edges: list[Edge] = Field(min_length=36, max_length=36)
    analysis_version: str
    window_size_back: int = Field(ge=1)
    affect_threshold: float = Field(ge=0, le=1)


def validate_artifacts(path: Path) -> dict:
    networks = {
        g: json.loads((path / f"{g}_network.json").read_text())
        for g in ["stable", "difficulty", "difference"]
    }
    stable, difficulty, difference = (
        networks[g] for g in ["stable", "difficulty", "difference"]
    )
    if stable["nodes"] != difficulty["nodes"] or stable["nodes"] != difference["nodes"]:
        raise ValueError("ENA coordinate mismatch")
    if {n["id"] for n in stable["nodes"]} != set(CODES):
        raise ValueError("ENA node vocabulary mismatch")
    for network in networks.values():
        Network.model_validate(network)
        pairs = {
            tuple(sorted((edge["source"], edge["target"]))) for edge in network["edges"]
        }
        if len(pairs) != 36 or any(
            a == b or a not in CODES or b not in CODES for a, b in pairs
        ):
            raise ValueError("Invalid or repeated ENA edge")
        if len(network["edges"]) != 36:
            raise ValueError("ENA edge count mismatch")
        if not all(np.isfinite(n[k]) for n in network["nodes"] for k in ["x", "y"]):
            raise ValueError("Invalid ENA coordinates")
    for a, b, d in zip(stable["edges"], difficulty["edges"], difference["edges"]):
        if (a["source"], a["target"]) != (b["source"], b["target"]) or (
            a["source"],
            a["target"],
        ) != (d["source"], d["target"]):
            raise ValueError("ENA edge ordering mismatch")
        if not np.isclose(b["weight"] - a["weight"], d["weight"], atol=1e-12):
            raise ValueError("Difference is not difficulty minus stable")
        if (
            not np.isfinite([d["weight"], d["ci_low"], d["ci_high"]]).all()
            or d["ci_low"] > d["ci_high"]
        ):
            raise ValueError("Invalid confidence interval")
    points = pl.read_csv(path / "unit_points.csv")
    if points["student_id"].n_unique() != points.height:
        raise ValueError("Repeated ENA unit")
    return {"passed": True, "units": points.height, "edges": 36}


def export_results(path: Path) -> dict:
    report = validate_artifacts(path)
    for name in ["unit_points", "unit_line_weights"]:
        pl.read_csv(path / f"{name}.csv").write_parquet(path / f"{name}.parquet")
    pl.read_csv(path / "difference_network.csv").write_parquet(
        path / "edge_statistics.parquet"
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="450" viewBox="0 0 1200 450"><rect width="1200" height="450" fill="#f8fafb"/>'
    ]
    for panel, group in enumerate(["stable", "difficulty", "difference"]):
        network = json.loads((path / f"{group}_network.json").read_text())
        nodes = network["nodes"]
        xmin, xmax = min(n["x"] for n in nodes), max(n["x"] for n in nodes)
        ymin, ymax = min(n["y"] for n in nodes), max(n["y"] for n in nodes)
        xy = {
            n["id"]: (
                panel * 400 + 50 + 300 * (n["x"] - xmin) / max(xmax - xmin, 1e-9),
                90 + 270 * (n["y"] - ymin) / max(ymax - ymin, 1e-9),
            )
            for n in nodes
        }
        parts.append(
            f'<text x="{panel * 400 + 30}" y="40" font-family="sans-serif" font-size="22">{group}</text>'
        )
        for edge in network["edges"]:
            x1, y1 = xy[edge["source"]]
            x2, y2 = xy[edge["target"]]
            color = "#c57547" if edge["weight"] < 0 else "#288d91"
            parts.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{max(0.2, abs(edge["weight"]) * 15)}"/>'
            )
        for name, (x, y) in xy.items():
            parts.append(
                f'<circle cx="{x}" cy="{y}" r="5" fill="#153a49"/><text x="{x}" y="{y - 9}" text-anchor="middle" font-family="sans-serif" font-size="9">{name}</text>'
            )
    parts.append("</svg>")
    (path / "plot.svg").write_text("".join(parts))
    (path / "plot.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>ENA networks</title><h1>Student-unit ENA</h1><p>Shared coordinates and edge widths. Co-occurrence does not imply causation.</p><img src="plot.svg" alt="Stable, difficulty and difference networks">'
    )
    (path / "validation.json").write_text(json.dumps(report, indent=2))
    return report
