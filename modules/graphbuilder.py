# modules/graphbuilder.py

import networkx as nx
from pyvis.network import Network
import pandas as pd

def build_graph_from_triples(triples_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from (subject, relation, object) triples."""
    G = nx.DiGraph()
    if triples_df is None or triples_df.empty:
        return G

    for _, row in triples_df.iterrows():
        s = str(row["subject"])
        o = str(row["object"])
        r = str(row.get("relation", ""))

        if not G.has_node(s):
            G.add_node(s)
        if not G.has_node(o):
            G.add_node(o)
        if G.has_edge(s, o):
            G[s][o]["weight"] += 1
        else:
            G.add_edge(s, o, label=r, weight=1)
    return G

def build_pyvis_html(G: nx.Graph) -> str:
    """Generate PyVis interactive HTML for a given NetworkX graph."""
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#020617",
        font_color="white",
    )
    net.from_nx(G)
    for u, v, data in G.edges(data=True):
        net.add_edge(
            u,
            v,
            title=data.get("label", ""),
            label=data.get("label", ""),
            value=data.get("weight", 1),
        )
    net.toggle_physics(True)
    html_str = net.generate_html()
    return html_str
