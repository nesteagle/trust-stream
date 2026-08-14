"""Earlier version of an internal thread visualization (legacy). 

Running on prior uncorrected data shows malformed future time references. 
Included for reference but not used in the final TrustStream system.
"""

import os
import textwrap
from collections import defaultdict, deque

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.pipeline.parser import load_comms_dataframe
from app.pipeline.preprocessing import resolve_and_tidy

AGENT_PALETTE = px.colors.qualitative.Plotly
EDGE_PALETTE = px.colors.qualitative.D3


def compute_graph_y(
    df_messages: pd.DataFrame,
    df_edges: pd.DataFrame,
) -> dict[str, float]:
    """Calculates Y-coordinates for DAG visualization of messages."""
    msg_df = df_messages.copy()
    msg_df["message_id"] = msg_df["message_id"].astype(str)

    all_msg_ids: set[str] = set(msg_df["message_id"])

    msg_metadata: dict[str, dict] = msg_df.set_index("message_id")[
        ["round_index", "timestamp"]
    ].to_dict("index")

    parents_map: dict[str, list[str]] = {mid: [] for mid in all_msg_ids}
    children_map: dict[str, list[str]] = {mid: [] for mid in all_msg_ids}

    edge_zip = zip(
        df_edges["message_id"].astype(str), df_edges["target_id"].astype(str)
    )
    for child, parent in edge_zip:
        if child not in all_msg_ids or parent not in all_msg_ids:
            continue
        parents_map[child].append(parent)
        children_map[parent].append(child)

    all_roots = [mid for mid in all_msg_ids if not parents_map[mid]]

    round_to_roots: dict[int, list[str]] = defaultdict(list)
    for root in all_roots:
        r_idx = msg_metadata.get(root, {}).get("round_index", 0)
        round_to_roots[r_idx].append(root)

    y_coords: dict[str, float] = {}
    global_visited: set[str] = set()

    intra_thread_spacing = 1.0
    inter_thread_buffer = 2.5
    round_separator_buffer = 6.0
    current_max_y = 0.0

    for round_idx in sorted(round_to_roots):
        roots_in_round = sorted(
            round_to_roots[round_idx],
            key=lambda mid: msg_metadata.get(mid, {}).get(
                "timestamp", pd.Timestamp.min
            ),
        )

        if current_max_y > 0.0:
            current_max_y += round_separator_buffer

        for root in roots_in_round:
            if root in global_visited:
                continue

            subtree_order: list[str] = []
            queue: deque[str] = deque([root])
            local_visited: set[str] = {root}

            while queue:
                node = queue.popleft()
                subtree_order.append(node)
                for kid in children_map.get(node, []):
                    if kid not in local_visited and kid not in global_visited:
                        local_visited.add(kid)
                        queue.append(kid)

            node_y: dict[str, float] = {}
            leaf_counter = 0

            for node in reversed(subtree_order):
                kids = [k for k in children_map.get(node, []) if k in local_visited]
                if not kids:
                    node_y[node] = leaf_counter * intra_thread_spacing
                    leaf_counter += 1
                else:
                    child_ys = [node_y[k] for k in kids if k in node_y]
                    if child_ys:
                        node_y[node] = (min(child_ys) + max(child_ys)) / 2.0
                    else:
                        node_y[node] = leaf_counter * intra_thread_spacing
                        leaf_counter += 1

            if node_y:
                min_ry = min(node_y.values())
                shift = current_max_y + inter_thread_buffer - min_ry

                for node, ry in node_y.items():
                    if node not in y_coords:
                        y_coords[node] = ry + shift
                        global_visited.add(node)

                current_max_y = max(node_y.values()) + shift

    return y_coords


def _build_hover_text(row: pd.Series) -> str:
    """Generates text for visualization mouse hover."""
    content = row["content"]
    agent = row["agent_role"]
    short_content = f"{content[:120]}…" if len(content) > 120 else content
    wrapped = "<br>".join(textwrap.wrap(short_content, width=50))

    return (
        f"<b>Agent:</b> {str(agent).upper()}<br>"
        f"<b>Round:</b> {row['round_index']}<br>"
        f"<b>Channel:</b> {row['channel']}<br>"
        f"<b>ID:</b> {row['message_id']}<br>"
        f"<b>Time:</b> {row['timestamp'].strftime('%H:%M:%S')}<br>"
        f"<br><i>{wrapped}</i>"
    )


def build_graph_visualization(
    df_messages: pd.DataFrame,
    df_edges: pd.DataFrame,
) -> go.Figure:
    """Generates and returns a Plotly graph object from layout data."""
    df_nodes = df_messages.copy()
    df_nodes["message_id"] = df_nodes["message_id"].astype(str)
    df_nodes["timestamp"] = pd.to_datetime(df_nodes["timestamp"])
    df_nodes = df_nodes.sort_values("timestamp").reset_index(drop=True)

    y_coords = compute_graph_y(df_nodes, df_edges)
    df_nodes["y_coord"] = df_nodes["message_id"].map(y_coords)

    node_lookup: dict[str, dict] = df_nodes.set_index("message_id").to_dict("index")
    traces: list[go.BaseTraceType] = []

    interaction_types = (
        df_edges["interaction_type"].unique().tolist()
        if "interaction_type" in df_edges.columns
        else []
    )
    edge_color_map = {
        itype: EDGE_PALETTE[i % len(EDGE_PALETTE)]
        for i, itype in enumerate(interaction_types)
    }
    edge_groups: dict[str, tuple[list, list]] = {
        itype: ([], []) for itype in interaction_types
    }

    edge_zip = zip(
        df_edges["message_id"].astype(str),
        df_edges["target_id"].astype(str),
        df_edges["interaction_type"].astype(str),
    )
    for child_id, parent_id, itype in edge_zip:
        if child_id not in node_lookup or parent_id not in node_lookup:
            continue
        if itype not in edge_groups:
            continue
        p = node_lookup[parent_id]
        c = node_lookup[child_id]

        xs, ys = edge_groups[itype]
        xs.extend([p["timestamp"], c["timestamp"], None])
        ys.extend([p["y_coord"], c["y_coord"], None])

    for itype, (xs, ys) in edge_groups.items():
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                line=dict(width=1.4, color=edge_color_map.get(itype, "#888888")),
                hoverinfo="none",
                mode="lines",
                name=itype,
                legendgroup="Interactions",
                legendgrouptitle=dict(text="Interaction type"),
                showlegend=True,
            )
        )

    agents = df_nodes["agent_role"].unique()
    channels = df_nodes["channel"].unique()

    agent_colors = {
        agent: AGENT_PALETTE[i % len(AGENT_PALETTE)] for i, agent in enumerate(agents)
    }

    _symbols = ["circle", "square", "star", "diamond", "cross", "x", "hexagon"]
    channel_symbols = {ch: _symbols[i % len(_symbols)] for i, ch in enumerate(channels)}

    for agent in agents:
        agent_df = df_nodes[df_nodes["agent_role"] == agent]
        hover_texts = agent_df.apply(_build_hover_text, axis=1).tolist()
        node_symbols = [channel_symbols.get(ch, "circle") for ch in agent_df["channel"]]

        traces.append(
            go.Scatter(
                x=agent_df["timestamp"],
                y=agent_df["y_coord"],
                mode="markers",
                name=str(agent),
                legendgroup="Agents",
                legendgrouptitle=dict(text="Agent (color)"),
                hoverinfo="text",
                text=hover_texts,
                marker=dict(
                    size=12,
                    color=agent_colors[agent],
                    symbol=node_symbols,
                    line=dict(width=1, color="#ffffff"),
                ),
            )
        )

    for ch in channels:
        traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=str(ch),
                legendgroup="Channels",
                legendgrouptitle=dict(text="Channel (shape)"),
                showlegend=True,
                marker=dict(
                    size=10,
                    color="#888888",
                    symbol=channel_symbols[ch],
                    line=dict(width=1, color="#ffffff"),
                ),
            )
        )

    return go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(
                text="Message Threads",
                font=dict(size=15, color="#444"),
            ),
            hovermode="closest",
            plot_bgcolor="#fafafa",
            paper_bgcolor="#ffffff",
            margin=dict(b=48, l=56, r=200, t=64),
            xaxis=dict(
                title="Simulation timeline",
                showgrid=True,
                gridcolor="#efefef",
                zeroline=False,
            ),
            yaxis=dict(
                title="Unique conversation thread",
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            legend=dict(
                itemsizing="constant",
                font=dict(size=11),
                groupclick="toggleitem",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#e0e0e0",
                borderwidth=1,
                x=1.01,
                xanchor="left",
                y=1.0,
                yanchor="top",
            ),
        ),
    )


def _save_visualization(fig: go.Figure, output_path: str) -> None:
    """Saves Plotly visualization to target path."""
    if dirname := os.path.dirname(output_path):
        os.makedirs(dirname, exist_ok=True)
    fig.write_html(output_path)
    print(f"Rendered DAG dashboard to: {output_path}")


if __name__ == "__main__":
    print("Loading data...")
    comms_df = load_comms_dataframe("data/raw/data.json")

    print("Tidying data, generating edges...")
    df_messages_clean, df_edges_clean = resolve_and_tidy(comms_df)

    print("Generating dashboard figure...")
    dag_fig = build_graph_visualization(df_messages_clean, df_edges_clean)

    print("Saving file out...")
    _save_visualization(dag_fig, output_path="output/threads.html")
