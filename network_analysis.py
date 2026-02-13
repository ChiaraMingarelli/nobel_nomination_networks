"""
Nobel Prize Nomination Network Analysis
========================================
Three network views:
1. Nominator -> Nominee directed graph (who champions whom)
2. Co-nomination network (nominees linked by shared nominators)
3. Country-level flow network (which countries nominate which)

Designed for integration into a Streamlit app.
Dependencies: pip install networkx pyvis pandas streamlit
"""

import networkx as nx
import pandas as pd
from pyvis.network import Network
from collections import defaultdict
import tempfile
import os

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


# ---------------------------------------------------------------------------
# 1. NOMINATOR -> NOMINEE GRAPH
# ---------------------------------------------------------------------------

def build_nomination_graph(df: pd.DataFrame,
                           category: str | None = None,
                           year_range: tuple[int, int] | None = None,
                           country: str | None = None) -> nx.DiGraph:
    """
    Directed graph: nominator -> nominee.
    Edge weight = number of times that nominator proposed that nominee.
    """
    filtered = _filter(df, category, year_range, country)

    G = nx.DiGraph()
    for (nominator, nominee), group in filtered.groupby(["nominator_name", "nominee_name"]):
        weight = len(group)
        years = sorted(group["year"].unique())
        G.add_edge(
            nominator, nominee,
            weight=weight,
            years=years,
        )
        # Node attributes
        if nominator not in G.nodes or "role" not in G.nodes[nominator]:
            row = group.iloc[0]
            G.nodes[nominator]["role"] = "nominator"
            G.nodes[nominator]["country"] = row.get("nominator_country", "Unknown") if "nominator_country" in row.index else "Unknown"
        G.nodes[nominee]["role"] = "nominee"
        G.nodes[nominee]["country"] = group.iloc[0].get("nominee_country", "Unknown") if "nominee_country" in group.columns else "Unknown"
        G.nodes[nominee]["total_nominations"] = filtered[
            filtered["nominee_name"] == nominee
        ].shape[0]
        if "nominee_prize_year" in group.columns:
            prize_year = group.iloc[0].get("nominee_prize_year")
            if pd.notna(prize_year):
                G.nodes[nominee]["prize_year"] = int(prize_year)

    return G


# ---------------------------------------------------------------------------
# 2. CO-NOMINATION NETWORK (nominees who share nominators)
# ---------------------------------------------------------------------------

def build_conomination_graph(df: pd.DataFrame,
                              category: str | None = None,
                              year_range: tuple[int, int] | None = None) -> nx.Graph:
    """
    Undirected graph among nominees.
    Edge weight = number of distinct nominators they share.
    This is the network that reveals "cultures of excellence" --
    clusters of nominees embedded in the same advocacy community.
    """
    filtered = _filter(df, category, year_range)

    # For each nominator, collect the set of nominees they proposed
    nominator_to_nominees = defaultdict(set)
    for _, row in filtered.iterrows():
        nominator_to_nominees[row["nominator_name"]].add(row["nominee_name"])

    # Build edges: for each nominator who proposed >1 nominee, link all pairs
    G = nx.Graph()
    edge_weights = defaultdict(int)
    for nominator, nominees in nominator_to_nominees.items():
        nominees = sorted(nominees)
        for i in range(len(nominees)):
            for j in range(i + 1, len(nominees)):
                edge_weights[(nominees[i], nominees[j])] += 1

    # Add nominee metadata
    nominee_countries = {}
    nominee_prize_years = {}
    nominee_counts = filtered.groupby("nominee_name").size().to_dict()
    for _, row in filtered.iterrows():
        if "nominee_country" in row.index:
            nominee_countries[row["nominee_name"]] = row.get("nominee_country", "Unknown")
        if "nominee_prize_year" in row.index:
            py = row.get("nominee_prize_year")
            if pd.notna(py):
                nominee_prize_years[row["nominee_name"]] = int(py)

    for (n1, n2), weight in edge_weights.items():
        G.add_edge(n1, n2, weight=weight)

    for node in G.nodes:
        G.nodes[node]["country"] = nominee_countries.get(node, "Unknown")
        G.nodes[node]["total_nominations"] = nominee_counts.get(node, 0)
        if node in nominee_prize_years:
            G.nodes[node]["prize_year"] = nominee_prize_years[node]

    return G


# ---------------------------------------------------------------------------
# 3. COUNTRY-LEVEL FLOW NETWORK
# ---------------------------------------------------------------------------

def build_country_flow_graph(df: pd.DataFrame,
                              category: str | None = None,
                              year_range: tuple[int, int] | None = None) -> nx.DiGraph:
    """
    Directed graph: nominator_country -> nominee_country.
    Edge weight = number of nominations along that route.
    Directly tests the paper's claim about domestic vs. foreign nomination patterns.
    """
    filtered = _filter(df, category, year_range)

    # Only include rows where both countries are known
    if "nominator_country" in filtered.columns and "nominee_country" in filtered.columns:
        filtered = filtered.dropna(subset=["nominator_country", "nominee_country"])
        filtered = filtered[
            (filtered["nominator_country"] != "Unknown") &
            (filtered["nominee_country"] != "Unknown")
        ]
    else:
        return nx.DiGraph()

    G = nx.DiGraph()
    for (src, dst), group in filtered.groupby(["nominator_country", "nominee_country"]):
        G.add_edge(src, dst, weight=len(group))

    return G


# ---------------------------------------------------------------------------
# CAMPAIGN DETECTION
# ---------------------------------------------------------------------------

def detect_campaigns(df: pd.DataFrame,
                     min_nominations: int = 5,
                     year_window: int = 3) -> pd.DataFrame:
    """
    Identify coordinated nomination campaigns:
    clusters of nominations for the same nominee within a short time window,
    especially from nominators at the same institution.

    Returns a DataFrame of suspected campaigns with stats.
    """
    filtered = df.copy()
    campaigns = []

    for nominee, group in filtered.groupby("nominee_name"):
        group = group.sort_values("year")
        years = group["year"].values

        if len(years) == 0:
            continue

        # Sliding window: find bursts
        for start_year in range(int(years.min()), int(years.max()) - year_window + 2):
            end_year = start_year + year_window - 1
            window = group[(group["year"] >= start_year) & (group["year"] <= end_year)]
            if len(window) >= min_nominations:
                nominators = window["nominator_name"].unique()
                campaigns.append({
                    "nominee": nominee,
                    "year_start": start_year,
                    "year_end": end_year,
                    "n_nominations": len(window),
                    "n_unique_nominators": len(nominators),
                    "nominators": list(nominators),
                })

    if not campaigns:
        return pd.DataFrame(columns=["nominee", "year_start", "year_end",
                                     "n_nominations", "n_unique_nominators", "nominators"])

    return pd.DataFrame(campaigns).sort_values("n_nominations", ascending=False)


# ---------------------------------------------------------------------------
# VISUALIZATION (pyvis for Streamlit)
# ---------------------------------------------------------------------------

COUNTRY_COLORS = {
    "USA": "#1f77b4",
    "Canada": "#d62728",
    "Germany": "#2ca02c",
    "France": "#9467bd",
    "UK": "#ff7f0e",
    "Sweden": "#e377c2",
    "Italy": "#8c564b",
    "the Netherlands": "#17becf",
    "Switzerland": "#bcbd22",
    "Austria": "#7f7f7f",
    "Denmark": "#aec7e8",
    "Russia": "#ff9896",
    "Japan": "#98df8a",
    "Unknown": "#cccccc",
}


def visualize_graph(G: nx.Graph | nx.DiGraph,
                    title: str = "Nobel Nomination Network",
                    height: str = "700px",
                    width: str = "100%",
                    size_by: str = "total_nominations",
                    color_by: str = "country",
                    min_edge_weight: int = 1,
                    physics: bool = True) -> str | None:
    """
    Render a networkx graph as an interactive pyvis HTML.
    Returns the path to the temp HTML file (for st.components.v1.html).
    """
    # Filter edges by minimum weight
    if min_edge_weight > 1:
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                           if d.get("weight", 1) < min_edge_weight]
        G = G.copy()
        G.remove_edges_from(edges_to_remove)
        # Remove isolated nodes after filtering
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)

    if len(G.nodes) == 0:
        return None

    directed = isinstance(G, nx.DiGraph)
    net = Network(height=height, width=width, directed=directed,
                  notebook=False, cdn_resources="remote")

    # Nodes
    for node in G.nodes:
        data = G.nodes[node]
        size = max(8, min(50, data.get(size_by, 1) * 2))
        color = COUNTRY_COLORS.get(data.get("country", ""), "#cccccc")
        label = node
        title_text = f"{node}\nCountry: {data.get('country', 'N/A')}"
        if "total_nominations" in data:
            title_text += f"\nTotal nominations: {data['total_nominations']}"
        if "prize_year" in data:
            title_text += f"\nWon: {data['prize_year']}"
        if "role" in data:
            title_text += f"\nRole: {data['role']}"

        net.add_node(node, label=label, size=size, color=color, title=title_text)

    # Edges
    for u, v, d in G.edges(data=True):
        weight = d.get("weight", 1)
        title_text = f"Weight: {weight}"
        if "years" in d:
            title_text += f"\nYears: {d['years']}"
        net.add_edge(u, v, value=weight, title=title_text)

    # Physics settings
    if physics:
        net.force_atlas_2based(gravity=-50, central_gravity=0.01,
                               spring_length=100, spring_strength=0.08)
    else:
        net.toggle_physics(False)

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w")
    net.save_graph(tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------------------------

def render_network_page(df: pd.DataFrame):
    """
    Full Streamlit page for network analysis.
    Category and year range are already filtered by the caller (sidebar).
    """
    if not HAS_STREAMLIT:
        print("Streamlit not available. Use this module within a Streamlit app.")
        return

    # --- Sidebar filters (below the data selection controls) ---
    st.sidebar.divider()
    st.sidebar.header("Network Controls")

    network_type = st.sidebar.selectbox("Network type", [
        "Nominator -> Nominee",
        "Co-nomination (shared nominators)",
    ])

    min_weight = st.sidebar.slider("Min edge weight", 1, 10, 1)

    country_filter = None
    if network_type == "Nominator -> Nominee" and "nominee_country" in df.columns:
        country_options = ["All"] + sorted(
            df["nominee_country"].dropna().unique().tolist()
        )
        country_filter = st.sidebar.selectbox("Filter nominee country", country_options)
        if country_filter == "All":
            country_filter = None

    # --- Main area ---
    st.header("Nomination Networks")

    # Build the appropriate graph
    if network_type == "Nominator -> Nominee":
        G = build_nomination_graph(df, country=country_filter)
        st.caption(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if G.number_of_nodes() > 0:
            stats = network_summary(G)
            cols = st.columns(len(stats))
            for i, (k, v) in enumerate(stats.items()):
                cols[i % len(cols)].metric(k, v)

    elif network_type == "Co-nomination (shared nominators)":
        G = build_conomination_graph(df)
        st.caption(f"{G.number_of_nodes()} nominees, {G.number_of_edges()} co-nomination links")

        if G.number_of_nodes() > 0:
            stats = network_summary(G)
            cols = st.columns(min(len(stats), 4))
            for i, (k, v) in enumerate(stats.items()):
                cols[i % len(cols)].metric(k, v)

        # Community detection
        if G.number_of_nodes() > 0 and st.checkbox("Run community detection (Louvain)"):
            try:
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(G, weight="weight", seed=42)
                for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:10]):
                    st.write(f"**Community {i+1}** ({len(comm)} members): {', '.join(sorted(comm)[:10])}{'...' if len(comm) > 10 else ''}")
            except ImportError:
                st.warning("Louvain requires networkx >= 2.8")

    # Description + legend
    if G.number_of_nodes() > 0:
        if network_type == "Nominator -> Nominee":
            st.markdown(
                "**Directed graph**: arrows point from nominator to nominee. "
                "Node size = total nominations received. "
                "Edge thickness = number of times that nominator proposed that nominee. "
                "Hover over nodes and edges for details."
            )
        elif network_type == "Co-nomination (shared nominators)":
            st.markdown(
                "**Undirected graph**: nominees are linked when they share nominators. "
                "Node size = total nominations received. "
                "Edge thickness = number of shared nominators. "
                "Clusters reveal communities of nominees championed by the same people."
            )
        # Color legend — only show countries present in the graph
        countries_in_graph = set()
        for node in G.nodes:
            c = G.nodes[node].get("country", "Unknown")
            if c:
                countries_in_graph.add(c)
        legend_items = []
        for country in sorted(countries_in_graph):
            color = COUNTRY_COLORS.get(country, "#cccccc")
            legend_items.append(
                f'<span style="color:{color}; font-size:20px;">&#9679;</span> {country}'
            )
        if legend_items:
            st.markdown(
                "**Node color = country:** " + " &nbsp;&nbsp; ".join(legend_items),
                unsafe_allow_html=True,
            )

    # Render interactive visualization
    if G.number_of_nodes() > 0:
        html_path = visualize_graph(G, title=network_type, min_edge_weight=min_weight)
        if html_path:
            with open(html_path, "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=720, scrolling=True)
            os.unlink(html_path)
        else:
            st.info("No edges meet the current filter criteria.")

    # Campaign detection (optional section)
    with st.expander("Campaign Detection"):
        min_noms = st.slider("Minimum nominations for campaign", 3, 15, 5, key="campaign_min")
        window = st.slider("Year window", 1, 5, 3, key="campaign_window")
        if st.button("Detect campaigns"):
            campaigns = detect_campaigns(df, min_nominations=min_noms, year_window=window)
            if len(campaigns) > 0:
                st.dataframe(campaigns[["nominee", "year_start", "year_end",
                                         "n_nominations", "n_unique_nominators"]],
                             hide_index=True)
            else:
                st.info("No campaigns detected with current thresholds.")

    # Raw data table
    with st.expander("Raw Edge Data"):
        st.dataframe(df, hide_index=True)


# ---------------------------------------------------------------------------
# NETWORK STATISTICS
# ---------------------------------------------------------------------------

def network_summary(G: nx.Graph | nx.DiGraph) -> dict:
    """Key network metrics for display."""
    stats = {
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": round(nx.density(G), 4),
    }

    if isinstance(G, nx.DiGraph):
        in_deg = sorted(G.in_degree(weight="weight"), key=lambda x: x[1], reverse=True)
        out_deg = sorted(G.out_degree(weight="weight"), key=lambda x: x[1], reverse=True)
        stats["Most nominated"] = f"{in_deg[0][0]} ({in_deg[0][1]})" if in_deg else "N/A"
        stats["Top nominator"] = f"{out_deg[0][0]} ({out_deg[0][1]})" if out_deg else "N/A"
    else:
        deg = sorted(G.degree(weight="weight"), key=lambda x: x[1], reverse=True)
        stats["Most connected"] = f"{deg[0][0]} ({deg[0][1]})" if deg else "N/A"

        # Connected components
        components = list(nx.connected_components(G))
        stats["Components"] = len(components)
        stats["Largest component"] = len(max(components, key=len)) if components else 0

    return stats


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _filter(df, category=None, year_range=None, country=None):
    filtered = df.copy()
    if category:
        filtered = filtered[filtered["category"] == category]
    if year_range:
        filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]
    if country and "nominee_country" in filtered.columns:
        filtered = filtered[filtered["nominee_country"] == country]
    return filtered
