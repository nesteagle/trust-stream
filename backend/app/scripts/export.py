import json

import networkx as nx
import numpy as np
import pandas as pd

from app.pipeline.evaluator import score_dataframe
from app.pipeline.parser import load_comms_dataframe
from app.pipeline.preprocessing import resolve_and_tidy


def export_for_frontend(
    df_messages: pd.DataFrame, df_edges: pd.DataFrame, output_path: str
):
    """Formats and exports the graph state for frontend."""
    # clean data
    df_nodes = df_messages.copy()
    df_nodes["timestamp"] = df_nodes["timestamp"].apply(
        lambda x: x.isoformat() if pd.notnull(x) else None
    )
    df_edges = df_edges.replace({np.nan: None})

    df_nodes["reasoning"] = (
        df_nodes[["reacting", "rationalizing", "deliberating"]].bfill(axis=1).iloc[:, 0]
    )
    df_nodes = df_nodes.replace({np.nan: None})

    # generate thread ID for frontend view
    G = nx.from_pandas_edgelist(
        df_edges, source="message_id", target="target_id", create_using=nx.DiGraph()
    )
    G.add_nodes_from(df_nodes["message_id"])
    nx.set_node_attributes(G, df_nodes.set_index("message_id").to_dict("index"))

    components = list(nx.weakly_connected_components(G))
    message_to_thread = {}
    thread_counter = 0

    valid_components = [c for c in components if len(c) > 1]
    for thread_counter, component in enumerate(valid_components):
        for msg_id in component:
            message_to_thread[msg_id] = thread_counter

    df_nodes["thread_id"] = df_nodes["message_id"].map(message_to_thread)
    df_nodes["thread_id"] = df_nodes["thread_id"].replace({np.nan: None})

    # group channels by visibility privacy
    channel_mapping = {
        "anonymous_post": "post",
        "comms_huddle": "internal",
        "official_post": "post",
        "one_on_one_chat": "private",
        "personal_post": "post",
        "side_huddle": "private",
    }

    df_nodes["visibility"] = (
        df_nodes["channel"].map(channel_mapping).fillna(df_nodes["channel"])
    )

    # filter + rename
    columns_to_keep = [
        "message_id",
        "timestamp",
        "agent_label",
        "content",
        "reasoning",
        "score_explanations_external",
        "score_explanations_internal",
        "thread_id",
        "channel",
        "visibility",
    ]
    nodes_records = (
        df_nodes[columns_to_keep]
        .rename(
            columns={
                "message_id": "id",
                "agent_label": "agent",
                "score_explanations_external": "explanationsExternal",
                "score_explanations_internal": "explanationsInternal",
                "thread_id": "threadId",
            }
        )
        .to_dict(orient="records")
    )

    edges_records = (
        df_edges[["message_id", "target_id"]]
        .rename(columns={"message_id": "source", "target_id": "target"})
        .to_dict(orient="records")
    )

    payload = {"nodes": nodes_records, "edges": edges_records}

    with open(output_path, "w") as f:
        json.dump(payload, f)

    print(f"Exported clean frontend payload to {output_path}")


def main():
    print("Loading data...")
    comms_df = load_comms_dataframe("data/raw/data.json")
    print("Tidying...")
    df_messages, df_edges = resolve_and_tidy(comms_df)
    print("Generating scores...")
    df_scored = score_dataframe(
        df_messages,
        eval_path="output/latest/eval_results_luna_median.json",
    )
    export_for_frontend(
        df_messages=df_scored,
        df_edges=df_edges,
        output_path="output/data_eval.json",
    )


if __name__ == "__main__":
    main()
