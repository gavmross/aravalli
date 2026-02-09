"""
Sankey diagram visualization for delinquency transition flows.

Functions:
    render_sankey_diagram — build an interactive Plotly Sankey from transition matrix data
"""

from typing import Union

import pandas as pd
import plotly.graph_objects as go


def _parse_rate(val: object) -> float:
    """Parse a rate value that may be float (0.42) or string ('42.07%')."""
    if isinstance(val, str):
        return float(val.strip().rstrip("%")) / 100
    return float(val)


def render_sankey_diagram(
    transition_data: Union[dict, pd.Series],
    total_loans: int,
    strata_label: str = "ALL",
) -> go.Figure:
    """
    Build a Plotly Sankey diagram from a single row of transition matrix output.

    Parameters
    ----------
    transition_data : dict or pd.Series
        One row from calculate_transition_matrix() output. Column values may be
        floats (0.42) or formatted strings ('42.07%').
    total_loans : int
        Total loan count for the row (used in subtitle).
    strata_label : str
        Label for the strata being displayed (e.g. 'ALL', 'Grade B').

    Returns
    -------
    go.Figure — Plotly Sankey diagram
    """
    d = dict(transition_data)

    # --- Parse conditional rates ---
    fp_clean = _parse_rate(d["from_current_to_fully_paid_clean"])
    curr_clean = _parse_rate(d["from_current_to_current_clean"])
    in_grace = _parse_rate(d["from_current_to_delinquent"])

    grace_still = _parse_rate(d["from_grace_still_in_grace"])
    grace_prog = _parse_rate(d["from_grace_progressed"])

    l16_cured = _parse_rate(d["from_late16_cured"])
    l16_still = _parse_rate(d["from_late16_still_in_late16"])
    l16_prog = _parse_rate(d["from_late16_progressed"])

    l31_still = _parse_rate(d["from_late31_still_in_late31"])
    l31_charged_off = _parse_rate(d["from_late31_charged_off"])

    # --- Compute absolute flows (fraction of total pool) ---
    # Column 2 → 3: conditional on In Grace Period population
    still_grace_abs = in_grace * grace_still
    late16_abs = in_grace * grace_prog

    # Column 3 → 4: conditional on Late (16-30) population
    cured_abs = late16_abs * l16_cured
    still_late16_abs = late16_abs * l16_still
    late31_abs = late16_abs * l16_prog

    # Column 4 → 5: conditional on Late (31-120) population
    still_late31_abs = late31_abs * l31_still
    charged_off_abs = late31_abs * l31_charged_off

    # --- Node definitions (11 nodes, 5 columns) ---
    # 0  Current              (Col 1)
    # 1  Fully Paid (Clean)   (Col 2)
    # 2  Current (Clean)      (Col 2)
    # 3  In Grace Period      (Col 2)
    # 4  Still in Grace       (Col 3)
    # 5  Late (16-30)         (Col 3)
    # 6  Cured                (Col 4)
    # 7  Still Late (16-30)   (Col 4)
    # 8  Late (31-120)        (Col 4)
    # 9  Still Late (31-120)  (Col 5)
    # 10 Charged Off          (Col 5)

    node_colors = [
        "#4A90D9",  # 0  Current
        "#2ECC71",  # 1  Fully Paid (Clean)
        "#27AE60",  # 2  Current (Clean)
        "#E67E22",  # 3  In Grace Period
        "#F39C12",  # 4  Still in Grace
        "#E74C3C",  # 5  Late (16-30)
        "#2ECC71",  # 6  Cured
        "#F39C12",  # 7  Still Late (16-30)
        "#C0392B",  # 8  Late (31-120)
        "#F39C12",  # 9  Still Late (31-120)
        "#922B21",  # 10 Charged Off
    ]

    node_pcts = [
        1.0,               # 0  Current = 100%
        fp_clean,          # 1  Fully Paid
        curr_clean,        # 2  Current Clean
        in_grace,          # 3  In Grace Period
        still_grace_abs,   # 4  Still in Grace
        late16_abs,        # 5  Late (16-30)
        cured_abs,         # 6  Cured
        still_late16_abs,  # 7  Still Late (16-30)
        late31_abs,        # 8  Late (31-120)
        still_late31_abs,  # 9  Still Late (31-120)
        charged_off_abs,   # 10 Charged Off
    ]

    node_names = [
        "Current",
        "Fully Paid (Clean)",
        "Current (Clean)",
        "In Grace Period",
        "Still in Grace",
        "Late (16-30)",
        "Cured",
        "Still Late (16-30)",
        "Late (31-120)",
        "Still Late (31-120)",
        "Charged Off",
    ]

    def _fmt_pct(v: float) -> str:
        if v >= 0.01:
            return f"{v * 100:.1f}%"
        if v >= 0.001:
            return f"{v * 100:.2f}%"
        return f"{v * 100:.3f}%"

    node_labels = [f"{name}\n{_fmt_pct(pct)}" for name, pct in zip(node_names, node_pcts)]

    # Fixed positions: 5 columns left-to-right
    node_x = [
        0.01,                       # Col 1: Current
        0.25, 0.25, 0.25,          # Col 2: Fully Paid, Current Clean, In Grace
        0.50, 0.50,                # Col 3: Still Grace, Late 16-30
        0.72, 0.72, 0.72,         # Col 4: Cured, Still Late 16-30, Late 31-120
        0.99, 0.99,                # Col 5: Still Late 31-120, Charged Off
    ]
    node_y = [
        0.50,                       # Col 1: Current
        0.12, 0.55, 0.93,          # Col 2
        0.85, 0.96,                # Col 3
        0.78, 0.90, 0.97,         # Col 4
        0.90, 0.98,                # Col 5
    ]

    # --- Link definitions (10 links) ---
    source = [0, 0, 0,  3, 3,  5, 5, 5,  8, 8]
    target = [1, 2, 3,  4, 5,  6, 7, 8,  9, 10]
    value = [
        fp_clean,           # Current → Fully Paid
        curr_clean,         # Current → Current Clean
        in_grace,           # Current → In Grace Period
        still_grace_abs,    # In Grace → Still in Grace
        late16_abs,         # In Grace → Late (16-30)
        cured_abs,          # Late (16-30) → Cured
        still_late16_abs,   # Late (16-30) → Still Late (16-30)
        late31_abs,         # Late (16-30) → Late (31-120)
        still_late31_abs,   # Late (31-120) → Still Late (31-120)
        charged_off_abs,    # Late (31-120) → Charged Off
    ]

    # Link colors = target node color at 25% opacity
    def _hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    link_colors = [_hex_to_rgba(node_colors[t]) for t in target]

    # --- Build figure ---
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=25,
            thickness=20,
            label=node_labels,
            color=node_colors,
            x=node_x,
            y=node_y,
            line=dict(width=0),
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f"Delinquency Transition Flow — {strata_label}"
                 f"<br><sup>{total_loans:,} loans</sup>",
            font=dict(size=16, color="#e2e8f0"),
        ),
        font=dict(
            family="DM Sans, sans-serif",
            size=12,
            color="#e2e8f0",
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig
