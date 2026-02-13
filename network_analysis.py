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
import numpy as np
import pandas as pd
from pyvis.network import Network
from collections import defaultdict
import tempfile
import os
import io
import math
import random

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

    # Pre-compute nomination counts per nominee (O(n) instead of O(n²))
    nominee_counts = filtered.groupby("nominee_name").size().to_dict()

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
        G.nodes[nominee]["total_nominations"] = nominee_counts.get(nominee, 0)
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
    for row in filtered.itertuples(index=False):
        nominator_to_nominees[row.nominator_name].add(row.nominee_name)

    # Build edges: for each nominator who proposed >1 nominee, link all pairs.
    # Cap at 50 nominees per nominator to avoid O(n^2) blowup for
    # institutional/committee nominators.
    MAX_NOMINEES_PER_NOMINATOR = 50
    G = nx.Graph()
    edge_weights = defaultdict(int)
    for nominator, nominees in nominator_to_nominees.items():
        nominees = sorted(nominees)
        if len(nominees) > MAX_NOMINEES_PER_NOMINATOR:
            continue  # skip institutional nominators
        for i in range(len(nominees)):
            for j in range(i + 1, len(nominees)):
                edge_weights[(nominees[i], nominees[j])] += 1

    # Add nominee metadata — use vectorized groupby instead of iterrows
    nominee_countries = {}
    nominee_prize_years = {}
    nominee_counts = filtered.groupby("nominee_name").size().to_dict()
    if "nominee_country" in filtered.columns:
        nominee_countries = (
            filtered.dropna(subset=["nominee_country"])
            .groupby("nominee_name")["nominee_country"]
            .first()
            .to_dict()
        )
    if "nominee_prize_year" in filtered.columns:
        prize_rows = filtered.dropna(subset=["nominee_prize_year"])
        if len(prize_rows) > 0:
            nominee_prize_years = (
                prize_rows.groupby("nominee_name")["nominee_prize_year"]
                .first()
                .astype(int)
                .to_dict()
            )

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

    Overlapping windows for the same nominee are merged: for each burst of
    overlapping qualifying windows, only the window with the maximum nomination
    count is reported.

    Returns a DataFrame of suspected campaigns with stats.
    """
    filtered = df.copy()
    campaigns = []

    for nominee, group in filtered.groupby("nominee_name"):
        group = group.sort_values("year")
        years = group["year"].values

        if len(years) == 0:
            continue

        # Sliding window: find all qualifying bursts
        raw_windows = []
        for start_year in range(int(years.min()), int(years.max()) - year_window + 2):
            end_year = start_year + year_window - 1
            window = group[(group["year"] >= start_year) & (group["year"] <= end_year)]
            if len(window) >= min_nominations:
                nominators = window["nominator_name"].unique()
                raw_windows.append({
                    "year_start": start_year,
                    "year_end": end_year,
                    "n_nominations": len(window),
                    "n_unique_nominators": len(nominators),
                    "nominators": list(nominators),
                })

        # Merge overlapping windows: group consecutive overlapping windows
        # and keep only the one with the max nomination count per burst.
        if not raw_windows:
            continue

        bursts = []
        current_burst = [raw_windows[0]]
        for w in raw_windows[1:]:
            # Overlapping if this window's start <= previous window's end
            if w["year_start"] <= current_burst[-1]["year_end"]:
                current_burst.append(w)
            else:
                bursts.append(current_burst)
                current_burst = [w]
        bursts.append(current_burst)

        for burst in bursts:
            best = max(burst, key=lambda w: w["n_nominations"])
            campaigns.append({
                "nominee": nominee,
                "year_start": best["year_start"],
                "year_end": best["year_end"],
                "n_nominations": best["n_nominations"],
                "n_unique_nominators": best["n_unique_nominators"],
                "nominators": best["nominators"],
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

        if color_by == "category":
            color = DISCIPLINE_COLORS.get(data.get("category", ""), "#999999")
        else:
            color = COUNTRY_COLORS.get(data.get("country", ""), "#cccccc")

        label = node
        title_text = f"{node}"
        if data.get("category"):
            title_text += f"\nCategory: {data['category']}"
        if data.get("country") and data["country"] != "Unknown":
            title_text += f"\nCountry: {data['country']}"
        if "total_nominations" in data:
            title_text += f"\nTotal nominations: {data['total_nominations']}"
        if "prize_year" in data:
            title_text += f"\nWon: {data['prize_year']}"
        if "role" in data:
            title_text += f"\nRole: {data['role']}"
        if data.get("is_laureate"):
            title_text += "\nLaureate"

        # Gold border for laureates
        node_opts = {"label": label, "size": size, "color": color, "title": title_text}
        if data.get("is_laureate"):
            node_opts["borderWidth"] = 3
            node_opts["color"] = {"background": color, "border": "#FFD700"}

        net.add_node(node, **node_opts)

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

@st.cache_resource
def _cached_build_graph(df_hash, _df, network_type, country_filter, _precomputed):
    """Cache graph construction — df_hash drives cache invalidation."""
    if network_type == "Nominator -> Nominee":
        return build_nomination_graph(_df, country=country_filter)
    elif network_type == "Co-nomination (shared nominators)":
        return build_conomination_graph(_df)
    elif network_type == "Cross-category Combined":
        return build_combined_nomination_graph(_df, _precomputed)
    return nx.Graph()


@st.cache_data
def _cached_visualize(graph_key, _G, title, min_edge_weight, color_by):
    """Cache PyVis HTML generation — graph_key drives cache invalidation."""
    html_path = visualize_graph(_G, title=title, min_edge_weight=min_edge_weight, color_by=color_by)
    if html_path:
        with open(html_path, "r") as f:
            html_content = f.read()
        os.unlink(html_path)
        return html_content
    return None


def _df_hash(df):
    """Deterministic hash for a DataFrame using hashlib to avoid collisions."""
    import hashlib
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    h.update(",".join(df.columns).encode())
    h.update(pd.util.hash_pandas_object(df).values.tobytes())
    return h.hexdigest()


def _render_graph_visualization(G, network_type, min_weight, is_cross_category):
    """Render the interactive PyVis graph and stats/legend below it."""
    if G.number_of_nodes() == 0:
        st.info("No data to display for current filters.")
        return

    color_mode = "category" if is_cross_category else "country"
    graph_key = (G.number_of_nodes(), G.number_of_edges(), network_type, min_weight, color_mode)
    html_content = _cached_visualize(graph_key, G, network_type, min_weight, color_mode)
    if html_content:
        st.components.v1.html(html_content, height=720, scrolling=True)
    else:
        st.info("No edges meet the current filter criteria.")

    # Network stats
    stats, name_stats = network_summary(G)
    cols = st.columns(len(stats))
    for i, (k, v) in enumerate(stats.items()):
        cols[i].metric(k, v)
    if name_stats:
        for label, name in name_stats.items():
            st.markdown(f"**{label}:** {name}")

    # Description
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
    elif is_cross_category:
        st.markdown(
            "**Undirected graph**: all 5 Nobel categories combined. "
            "Nominator-nominee pairs are linked. "
            "Node color = discipline. Gold border = laureate. "
            "Hover over nodes for details."
        )

    # Color legend
    if is_cross_category:
        categories_in_graph = set()
        for node in G.nodes:
            c = G.nodes[node].get("category", "")
            if c:
                categories_in_graph.add(c)
        legend_items = []
        for cat in sorted(categories_in_graph):
            color = DISCIPLINE_COLORS.get(cat, "#999999")
            legend_items.append(
                f'<span style="color:{color}; font-size:20px;">&#9679;</span> {cat}'
            )
        legend_items.append(
            '<span style="color:#FFD700; font-size:20px;">&#9679;</span> Laureate (gold border)'
        )
        st.markdown(
            "**Node color = discipline:** " + " &nbsp;&nbsp; ".join(legend_items),
            unsafe_allow_html=True,
        )
    else:
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

    # Community detection (co-nomination only)
    if network_type == "Co-nomination (shared nominators)":
        if st.checkbox("Run community detection (Louvain)"):
            try:
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(G, weight="weight", seed=42)
                for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:10]):
                    st.write(f"**Community {i+1}** ({len(comm)} members): {', '.join(sorted(comm)[:10])}{'...' if len(comm) > 10 else ''}")
            except ImportError:
                st.warning("Louvain requires networkx >= 2.8")


def _render_campaigns(df, min_noms, campaign_window):
    """Render campaign detection results."""
    st.subheader("Campaign Detection")
    campaigns = detect_campaigns(df, min_nominations=min_noms, year_window=campaign_window)
    if len(campaigns) > 0:
        display_cols = ["nominee", "year_start", "year_end",
                        "n_nominations", "n_unique_nominators"]
        st.dataframe(campaigns[display_cols], hide_index=True)
        _csv_download_button(
            campaigns[display_cols],
            "campaigns.csv", key="campaigns_csv",
        )
    else:
        st.info("No campaigns detected with current thresholds.")


def _render_paper_analyses(df, combined_df, precomputed, has_combined, run_lcc):
    """Render the Gallotti & De Domenico paper analyses (endorsement + LCC)."""
    import matplotlib.pyplot as plt

    st.subheader("Paper Analyses (Gallotti & De Domenico, 2019)")
    st.caption(
        "Inspired by *Effects of homophily and academic reputation "
        "in the nomination and selection of Nobel laureates*."
    )

    # --- Endorsement Effect ---
    st.markdown("#### Laureate Endorsement Effect")
    st.caption("Do nominees endorsed by past laureates win at higher rates?")
    analysis_df = combined_df if has_combined else df
    effect = compute_endorsement_effect(analysis_df, precomputed)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Endorsed win rate",
        f"{effect['endorsed_rate']:.1%}",
        help=f"{effect['endorsed_won']}/{effect['endorsed_total']} nominees endorsed by a past laureate went on to win",
    )
    col2.metric(
        "Non-endorsed win rate",
        f"{effect['not_endorsed_rate']:.1%}",
        help=f"{effect['not_endorsed_won']}/{effect['not_endorsed_total']} nominees without laureate endorsement went on to win",
    )
    ratio_str = f"{effect['ratio']:.1f}x" if effect['ratio'] != float("inf") else "N/A"
    col3.metric("Ratio", ratio_str)

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(
        ["Endorsed\nby laureate", "Not endorsed"],
        [effect["endorsed_rate"] * 100, effect["not_endorsed_rate"] * 100],
        color=["#FFD700", "#999999"],
        edgecolor="black",
    )
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Laureate endorsement effect")
    max_val = max(effect["endorsed_rate"], effect["not_endorsed_rate"]) * 100
    ax.set_ylim(0, max_val * 1.2)
    for bar, rate in zip(bars, [effect["endorsed_rate"], effect["not_endorsed_rate"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    _fig_download_buttons(fig, "endorsement_effect", "endorsement")
    plt.close(fig)

    # --- Laureates in LCC (results in sidebar) ---
    if run_lcc:
        with st.sidebar:
            with st.spinner("Running LCC permutation test..."):
                if has_combined:
                    G_combined = build_combined_nomination_graph(combined_df, precomputed)
                else:
                    G_combined = build_combined_nomination_graph(df, precomputed)
                lcc_result = compute_lcc_analysis(G_combined, precomputed)

            if "error" in lcc_result:
                st.error(lcc_result["error"])
            else:
                st.markdown("**LCC Results**")
                st.metric("Observed in LCC", lcc_result["observed"],
                          help=f"Out of {lcc_result['n_laureates']} laureates")
                st.metric("Expected (null)", f"{lcc_result['expected_mean']} +/- {lcc_result['expected_std']}")
                st.metric("Z-score", lcc_result["z_score"],
                          help=f"p = {lcc_result['p_value']:.2e}")
                st.caption(
                    f"LCC: {lcc_result['lcc_size']}/{lcc_result['graph_size']} nodes. "
                    f"Laureates: {lcc_result['n_laureates']}."
                )


def render_network_page(df: pd.DataFrame, precomputed: dict | None = None,
                        combined_df: pd.DataFrame | None = None):
    """
    Full Streamlit page for network analysis.
    Category and year range are already filtered by the caller (sidebar).

    Delegates to helper functions for each section:
      _render_graph_visualization, _render_campaigns,
      _render_paper_analyses, _render_advanced_analyses
    """
    if not HAS_STREAMLIT:
        print("Streamlit not available. Use this module within a Streamlit app.")
        return

    has_combined = combined_df is not None and not combined_df.empty
    has_precomputed = precomputed is not None and len(precomputed) > 0

    # --- Sidebar: Network Controls ---
    st.sidebar.divider()
    st.sidebar.header("Network Controls")

    network_options = [
        "Nominator -> Nominee",
        "Co-nomination (shared nominators)",
    ]
    if has_combined:
        network_options.append("Cross-category Combined")

    network_type = st.sidebar.selectbox("Network type", network_options)
    is_cross_category = network_type == "Cross-category Combined"

    default_weight = 2 if is_cross_category else 1
    min_weight = st.sidebar.slider("Min edge weight", 1, 10, default_weight)

    if is_cross_category and min_weight == 1:
        st.sidebar.warning("Min weight 1 produces ~15K nodes — rendering may be slow.")

    country_filter = None
    if network_type == "Nominator -> Nominee" and "nominee_country" in df.columns:
        country_options = ["All"] + sorted(
            df["nominee_country"].dropna().unique().tolist()
        )
        country_filter = st.sidebar.selectbox("Filter nominee country", country_options)
        if country_filter == "All":
            country_filter = None

    # --- Sidebar: Analysis options ---
    st.sidebar.divider()
    st.sidebar.header("Analyses")

    show_campaigns = st.sidebar.checkbox("Campaign Detection", key="show_campaigns")
    run_campaigns = False
    if show_campaigns:
        min_noms = st.sidebar.slider("Min nominations", 3, 15, 5, key="campaign_min")
        campaign_window = st.sidebar.slider("Year window", 1, 5, 3, key="campaign_window")
        run_campaigns = st.sidebar.button("Detect campaigns")

    show_paper = has_precomputed and st.sidebar.checkbox(
        "Paper Analyses (Gallotti & De Domenico)", key="show_paper",
    )
    run_lcc = False
    if show_paper:
        run_lcc = st.sidebar.button("Run LCC Analysis", key="lcc_btn")
    show_raw = st.sidebar.checkbox("Raw Edge Data", key="show_raw")

    # --- Sidebar: Advanced Analyses ---
    st.sidebar.divider()
    st.sidebar.header("Advanced Analyses")

    adv_options = ["None"]
    if has_combined:
        adv_options.append("Temporal Evolution")
    if has_combined and has_precomputed:
        adv_options.extend([
            "Three Degrees of Influence",
            "Near-Miss Analysis",
            "Centrality Predicts Winners",
        ])

    adv_selection = st.sidebar.selectbox(
        "Select analysis", adv_options, key="adv_analysis_select")

    show_centrality = adv_selection == "Centrality Predicts Winners"
    show_near_miss = adv_selection == "Near-Miss Analysis"
    show_temporal = adv_selection == "Temporal Evolution"
    show_proximity = adv_selection == "Three Degrees of Influence"

    run_centrality = run_near_miss = run_temporal = run_proximity = False
    near_miss_min = 10

    if show_centrality:
        run_centrality = st.sidebar.button("Run Centrality Analysis", key="centrality_btn")
    elif show_near_miss:
        near_miss_min = st.sidebar.slider("Min nominations", 5, 30, 10, key="near_miss_min")
        run_near_miss = st.sidebar.button("Run Near-Miss Analysis", key="near_miss_btn")
    elif show_temporal:
        run_temporal = st.sidebar.button("Run Evolution Analysis", key="temporal_btn")
    elif show_proximity:
        run_proximity = st.sidebar.button("Run Proximity Analysis", key="proximity_btn")

    # --- Main area: Graph visualization ---
    st.header("Nomination Networks")

    if network_type == "Nominator -> Nominee":
        G = _cached_build_graph(_df_hash(df), df, network_type, country_filter, None)
    elif network_type == "Co-nomination (shared nominators)":
        G = _cached_build_graph(_df_hash(df), df, network_type, None, None)
    elif is_cross_category:
        with st.spinner("Building cross-category combined network..."):
            G = _cached_build_graph(_df_hash(combined_df), combined_df, network_type, None, precomputed)

    _render_graph_visualization(G, network_type, min_weight, is_cross_category)

    # --- Main area: Analyses ---
    if show_campaigns and run_campaigns:
        _render_campaigns(df, min_noms, campaign_window)

    if show_paper:
        _render_paper_analyses(df, combined_df, precomputed, has_combined, run_lcc)

    if show_raw:
        st.subheader("Raw Edge Data")
        st.dataframe(df, hide_index=True)
        _csv_download_button(df, "nomination_edges.csv", key="raw_edges_csv")

    if has_combined and adv_selection != "None":
        adv_flags = {
            "show_centrality": show_centrality,
            "run_centrality": run_centrality,
            "show_near_miss": show_near_miss,
            "run_near_miss": run_near_miss,
            "near_miss_min": near_miss_min,
            "show_temporal": show_temporal,
            "run_temporal": run_temporal,
            "show_proximity": show_proximity,
            "run_proximity": run_proximity,
        }
        st.divider()
        st.header("Advanced Network Analyses")
        _render_advanced_analyses(combined_df, precomputed, adv_flags)


# ---------------------------------------------------------------------------
# NETWORK STATISTICS
# ---------------------------------------------------------------------------

def network_summary(G: nx.Graph | nx.DiGraph) -> tuple[dict, dict]:
    """Key network metrics for display.

    Returns (stats, name_stats) where stats are numeric metrics
    suitable for st.metric and name_stats are name-based entries
    displayed as markdown to avoid truncation.
    """
    stats = {
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": round(nx.density(G), 4),
    }
    name_stats = {}

    if isinstance(G, nx.DiGraph):
        in_deg = sorted(G.in_degree(weight="weight"), key=lambda x: x[1], reverse=True)
        out_deg = sorted(G.out_degree(weight="weight"), key=lambda x: x[1], reverse=True)
        if in_deg:
            name_stats["Most nominated"] = f"{in_deg[0][0]} ({int(in_deg[0][1])})"
        if out_deg:
            name_stats["Top nominator"] = f"{out_deg[0][0]} ({int(out_deg[0][1])})"
    else:
        deg = sorted(G.degree(weight="weight"), key=lambda x: x[1], reverse=True)
        if deg:
            name_stats["Most connected"] = f"{deg[0][0]} ({int(deg[0][1])})"

        # Connected components
        components = list(nx.connected_components(G))
        stats["Components"] = len(components)
        stats["Largest component"] = len(max(components, key=len)) if components else 0

    return stats, name_stats


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


def _fig_download_buttons(fig, filename_stem: str, key_prefix: str):
    """Render PDF and PNG download buttons for a matplotlib figure."""
    col_pdf, col_png, _ = st.columns([1, 1, 3])
    # PDF
    buf_pdf = io.BytesIO()
    try:
        fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
        col_pdf.download_button(
            "Download PDF", buf_pdf.getvalue(),
            file_name=f"{filename_stem}.pdf", mime="application/pdf",
            key=f"{key_prefix}_pdf",
        )
    except ValueError:
        # PDF backend not available — skip
        pass
    # PNG
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
    col_png.download_button(
        "Download PNG", buf_png.getvalue(),
        file_name=f"{filename_stem}.png", mime="image/png",
        key=f"{key_prefix}_png",
    )


def _csv_download_button(dataframe: pd.DataFrame, filename: str, key: str,
                         label: str = "Download CSV"):
    """Render a CSV download button for a DataFrame."""
    csv_data = dataframe.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv_data, file_name=filename,
                       mime="text/csv", key=key)


# ---------------------------------------------------------------------------
# CATEGORY COLORS (for cross-category view)
# ---------------------------------------------------------------------------

DISCIPLINE_COLORS = {
    "Physics": "#1f77b4",
    "Chemistry": "#2ca02c",
    "Physiology or Medicine": "#d62728",
    "Medicine": "#d62728",
    "Literature": "#9467bd",
    "Peace": "#ff7f0e",
    "nominator": "#999999",
}


# ---------------------------------------------------------------------------
# NAME MATCHING (for linking nominators to laureates)
# ---------------------------------------------------------------------------

def normalize_laureate_name(name: str) -> str:
    """Lowercase, strip middle initials (1-2 char tokens), remove title suffixes."""
    name = name.lower().strip()
    # Remove common suffixes
    for suffix in [", jr.", ", jr", " jr.", " jr", ", sr.", ", sr", " sr.", " sr"]:
        name = name.replace(suffix, "")
    tokens = name.split()
    # Keep only tokens longer than 2 chars (drop middle initials like "A", "C", "von")
    # But keep 'von', 'de', 'van' etc. — only drop single-letter tokens and 2-letter with period
    filtered = []
    for t in tokens:
        if len(t) <= 2 and (len(t) == 1 or t.endswith(".")):
            continue
        filtered.append(t)
    return " ".join(filtered)


def build_laureate_lookup(precomputed: dict) -> dict:
    """
    Build {normalized_name: {name, year_won, category, id}} for all laureates
    across all 5 categories in precomputed_stats.
    """
    lookup = {}
    for cat_key, entries in precomputed.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            name = entry.get("Name", "")
            year_won = entry.get("Year Won")
            category = entry.get("Prize Category", cat_key)
            eid = str(entry.get("ID", ""))
            if not name or not year_won:
                continue
            norm = normalize_laureate_name(name)
            lookup[norm] = {
                "name": name,
                "year_won": int(year_won),
                "category": category,
                "id": eid,
            }
    return lookup


def match_nominator_to_laureate(nominator_name: str, laureate_lookup: dict) -> dict | None:
    """
    Two-tier matching: (1) full normalized match, (2) first+last name fallback.
    Returns laureate info dict or None.
    """
    norm = normalize_laureate_name(nominator_name)

    # Tier 1: exact normalized match
    if norm in laureate_lookup:
        return laureate_lookup[norm]

    # Tier 2: first + last name only
    tokens = norm.split()
    if len(tokens) >= 2:
        short = f"{tokens[0]} {tokens[-1]}"
        for key, val in laureate_lookup.items():
            key_tokens = key.split()
            if len(key_tokens) >= 2:
                key_short = f"{key_tokens[0]} {key_tokens[-1]}"
                if short == key_short:
                    return val

    return None


# ---------------------------------------------------------------------------
# ANALYSIS 1: LAUREATE ENDORSEMENT EFFECT
# ---------------------------------------------------------------------------

def compute_endorsement_effect(df: pd.DataFrame, precomputed: dict) -> dict:
    """
    For each nominee, check whether any of their nominators were laureates
    who won strictly before the nomination year. Compare win rates.

    Returns dict with endorsed/non-endorsed win rates and ratio.
    """
    laureate_lookup = build_laureate_lookup(precomputed)

    # For each unique nominee, track: endorsed by a laureate? did they win?
    nominee_info = {}  # nominee_name -> {endorsed: bool, won: bool}

    has_prize_col = "nominee_prize_year" in df.columns
    for row in df.itertuples(index=False):
        nominee = row.nominee_name
        nom_year = row.year
        nominator = row.nominator_name

        if nominee not in nominee_info:
            won = has_prize_col and pd.notna(row.nominee_prize_year)
            nominee_info[nominee] = {"endorsed": False, "won": bool(won)}

        # Check if this nominator was a laureate who won before this nomination
        laureate = match_nominator_to_laureate(nominator, laureate_lookup)
        if laureate and laureate["year_won"] < nom_year:
            nominee_info[nominee]["endorsed"] = True

    endorsed_won = sum(1 for v in nominee_info.values() if v["endorsed"] and v["won"])
    endorsed_total = sum(1 for v in nominee_info.values() if v["endorsed"])
    not_endorsed_won = sum(1 for v in nominee_info.values() if not v["endorsed"] and v["won"])
    not_endorsed_total = sum(1 for v in nominee_info.values() if not v["endorsed"])

    endorsed_rate = endorsed_won / endorsed_total if endorsed_total > 0 else 0
    not_endorsed_rate = not_endorsed_won / not_endorsed_total if not_endorsed_total > 0 else 0
    ratio = endorsed_rate / not_endorsed_rate if not_endorsed_rate > 0 else float("inf")

    return {
        "endorsed_total": endorsed_total,
        "endorsed_won": endorsed_won,
        "endorsed_rate": endorsed_rate,
        "not_endorsed_total": not_endorsed_total,
        "not_endorsed_won": not_endorsed_won,
        "not_endorsed_rate": not_endorsed_rate,
        "ratio": ratio,
    }


# ---------------------------------------------------------------------------
# ANALYSIS 2: CROSS-CATEGORY COMBINED NETWORK
# ---------------------------------------------------------------------------

def build_combined_nomination_graph(df: pd.DataFrame, precomputed: dict) -> nx.Graph:
    """
    Build an undirected graph: nominator <-> nominee edges, accumulate weight.
    Node attributes include category (from laureate data or most-frequent
    nomination category) and is_laureate flag.
    """
    laureate_lookup = build_laureate_lookup(precomputed)

    G = nx.Graph()
    edge_weights = defaultdict(int)
    node_categories = defaultdict(lambda: defaultdict(int))  # node -> {category: count}

    for row in df.itertuples(index=False):
        nominator = row.nominator_name
        nominee = row.nominee_name
        cat = getattr(row, "category", "Unknown") or "Unknown"

        edge_weights[(nominator, nominee)] += 1
        node_categories[nominee][cat] += 1
        node_categories[nominator][cat] += 1

    for (n1, n2), weight in edge_weights.items():
        G.add_edge(n1, n2, weight=weight)

    # Set node attributes
    for node in G.nodes:
        # Check if this node is a laureate
        laureate = match_nominator_to_laureate(node, laureate_lookup)
        if laureate:
            G.nodes[node]["category"] = laureate["category"]
            G.nodes[node]["is_laureate"] = True
            G.nodes[node]["prize_year"] = laureate["year_won"]
        else:
            # Use most-frequent nomination category
            cats = node_categories.get(node, {})
            if cats:
                G.nodes[node]["category"] = max(cats, key=cats.get)
            else:
                G.nodes[node]["category"] = "Unknown"
            G.nodes[node]["is_laureate"] = False

        G.nodes[node]["total_nominations"] = sum(
            d.get("weight", 1) for _, _, d in G.edges(node, data=True)
        )

    return G


# ---------------------------------------------------------------------------
# ANALYSIS 3: LAUREATES IN LARGEST CONNECTED COMPONENT
# ---------------------------------------------------------------------------

def compute_lcc_analysis(G: nx.Graph, precomputed: dict, n_permutations: int = 1000) -> dict:
    """
    Test whether laureates are over-represented in the largest connected component.

    Counts unique laureate IDs (not nodes) to avoid inflating the count when the
    same person appears under multiple name spellings.

    Null model: randomly sample n_laureates from *nominee nodes only* (nodes
    that appear as nominees, i.e. have the "role" attribute set to "nominee" or
    appear in the nominee column). Laureates are nominees, not random nodes, so
    sampling from all nodes would undercount the expected LCC overlap.

    Returns dict with observed count, expected mean/std, z-score, and sizes.
    """
    laureate_lookup = build_laureate_lookup(precomputed)

    # Map each node to its laureate ID (if any), deduplicating by ID
    node_to_lid = {}
    for node in G.nodes:
        laureate = match_nominator_to_laureate(node, laureate_lookup)
        if laureate:
            node_to_lid[node] = laureate["id"]

    # Count unique laureate IDs
    unique_lids = set(node_to_lid.values())
    n_laureates = len(unique_lids)
    if n_laureates == 0:
        return {"error": "No laureates found in graph"}

    # Find largest connected component
    components = list(nx.connected_components(G))
    if not components:
        return {"error": "No connected components"}

    lcc = max(components, key=len)
    lcc_size = len(lcc)
    graph_size = G.number_of_nodes()

    # Observed: how many unique laureate IDs have at least one node in the LCC?
    lids_in_lcc = {node_to_lid[n] for n in node_to_lid if n in lcc}
    observed = len(lids_in_lcc)

    # Build nominee-only node pool for the null model.
    # Nominees are identified by role attribute (set by build_nomination_graph)
    # or by is_laureate attribute (set by build_combined_nomination_graph).
    # Fallback: any node that has total_nominations > 0 or is a laureate.
    nominee_nodes = []
    for node in G.nodes:
        data = G.nodes[node]
        if data.get("role") == "nominee":
            nominee_nodes.append(node)
        elif data.get("is_laureate"):
            nominee_nodes.append(node)
        elif data.get("total_nominations", 0) > 0:
            nominee_nodes.append(node)

    # Ensure pool is at least as large as the number of laureates
    if len(nominee_nodes) < n_laureates:
        nominee_nodes = list(G.nodes)  # fall back to all nodes

    lcc_set = set(lcc)
    rng = random.Random(42)
    null_counts = []
    for _ in range(n_permutations):
        random_picks = set(rng.sample(nominee_nodes, min(n_laureates, len(nominee_nodes))))
        null_counts.append(len(random_picks & lcc_set))

    expected_mean = sum(null_counts) / len(null_counts)
    variance = sum((x - expected_mean) ** 2 for x in null_counts) / len(null_counts)
    expected_std = math.sqrt(variance) if variance > 0 else 1e-10

    z_score = (observed - expected_mean) / expected_std
    # One-sided p-value using complementary error function
    p_value = math.erfc(z_score / math.sqrt(2)) / 2

    return {
        "observed": observed,
        "expected_mean": round(expected_mean, 1),
        "expected_std": round(expected_std, 2),
        "z_score": round(z_score, 2),
        "p_value": p_value,
        "lcc_size": lcc_size,
        "graph_size": graph_size,
        "n_laureates": n_laureates,
        "null_pool_size": len(nominee_nodes),
    }


# ---------------------------------------------------------------------------
# ANALYSIS 4: TEMPORAL NETWORK EVOLUTION
# ---------------------------------------------------------------------------

class _UnionFind:
    """Incremental union-find for tracking connected component sizes."""

    def __init__(self):
        self._parent = {}
        self._rank = {}
        self._size = {}  # size of component rooted at each root
        self._max_size = 0
        self._n_components = 0

    def add(self, x):
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            self._size[x] = 1
            self._n_components += 1
            if 1 > self._max_size:
                self._max_size = 1

    def find(self, x):
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x, y):
        self.add(x)
        self.add(y)
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        self._size[rx] += self._size[ry]
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        self._n_components -= 1
        if self._size[rx] > self._max_size:
            self._max_size = self._size[rx]

    @property
    def n_nodes(self):
        return len(self._parent)

    @property
    def n_components(self):
        return self._n_components

    @property
    def gcc_size(self):
        return self._max_size


def compute_temporal_evolution(combined_df: pd.DataFrame) -> dict:
    """
    How do network properties evolve year by year?

    Builds cumulative undirected nomination graphs incrementally from 1901
    to 1970 and computes structural metrics at each year. Uses an incremental
    union-find for O(alpha(n)) connected-component tracking instead of
    recomputing components from scratch each year.

    Returns dict with:
      - overall: DataFrame indexed by year with network metrics
      - by_category: dict {category: DataFrame} with per-category metrics
      - category_list: list of categories present
    """
    df = combined_df.copy()
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    min_year = max(1901, df["year"].min())
    max_year = min(1970, df["year"].max())
    years = list(range(min_year, max_year + 1))

    categories = sorted(df["category"].dropna().unique())

    # --- Overall cumulative graph ---
    G = nx.Graph()
    uf = _UnionFind()
    overall_rows = []
    edges_by_year = df.groupby("year")

    for yr in years:
        if yr in edges_by_year.groups:
            year_df = edges_by_year.get_group(yr)
            for row in year_df.itertuples(index=False):
                n1, n2 = row.nominator_name, row.nominee_name
                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += 1
                else:
                    G.add_edge(n1, n2, weight=1)
                uf.union(n1, n2)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_nodes > 0:
            gcc_size = uf.gcc_size
            gcc_frac = gcc_size / n_nodes
            n_components = uf.n_components
            mean_degree = 2 * n_edges / n_nodes
            # Clustering is expensive for large graphs — sample for speed
            if n_nodes > 5000:
                sample_nodes = random.sample(list(G.nodes), min(2000, n_nodes))
                clustering = nx.average_clustering(G, nodes=sample_nodes)
            else:
                clustering = nx.average_clustering(G)
        else:
            gcc_size = gcc_frac = clustering = mean_degree = 0
            n_components = 0

        overall_rows.append({
            "year": yr,
            "nodes": n_nodes,
            "edges": n_edges,
            "gcc_size": gcc_size,
            "gcc_frac": round(gcc_frac, 4),
            "clustering": round(clustering, 4),
            "mean_degree": round(mean_degree, 3),
            "n_components": n_components,
        })

    overall = pd.DataFrame(overall_rows).set_index("year")

    # --- Per-category cumulative graphs ---
    by_category = {}
    for cat in categories:
        cat_df = df[df["category"] == cat]
        cat_edges_by_year = cat_df.groupby("year")
        G_cat = nx.Graph()
        uf_cat = _UnionFind()
        cat_rows = []

        for yr in years:
            if yr in cat_edges_by_year.groups:
                year_df = cat_edges_by_year.get_group(yr)
                for row in year_df.itertuples(index=False):
                    n1, n2 = row.nominator_name, row.nominee_name
                    if G_cat.has_edge(n1, n2):
                        G_cat[n1][n2]["weight"] += 1
                    else:
                        G_cat.add_edge(n1, n2, weight=1)
                    uf_cat.union(n1, n2)

            n_nodes = G_cat.number_of_nodes()
            n_edges = G_cat.number_of_edges()

            if n_nodes > 1:
                gcc_size = uf_cat.gcc_size
                gcc_frac = gcc_size / n_nodes
                mean_degree = 2 * n_edges / n_nodes
            elif n_nodes == 1:
                gcc_size = 1
                gcc_frac = 1.0
                mean_degree = 0
            else:
                gcc_size = gcc_frac = mean_degree = 0

            cat_rows.append({
                "year": yr,
                "nodes": n_nodes,
                "edges": n_edges,
                "gcc_size": gcc_size,
                "gcc_frac": round(gcc_frac, 4),
                "mean_degree": round(mean_degree, 3),
            })

        by_category[cat] = pd.DataFrame(cat_rows).set_index("year")

    return {
        "overall": overall,
        "by_category": by_category,
        "category_list": categories,
    }


# ---------------------------------------------------------------------------
# ANALYSIS 5: THREE DEGREES OF INFLUENCE (PROXIMITY EFFECT)
# ---------------------------------------------------------------------------

def compute_proximity_effect(combined_df: pd.DataFrame, precomputed: dict) -> dict:
    """
    In the co-nomination network, does proximity to a past laureate predict winning?

    Builds undirected co-nomination graph, computes shortest-path distances from
    each non-laureate nominee to the nearest laureate who won before the nominee's
    first nomination year. Uses first (not last) year to avoid time-period bias:
    early nominees have few past laureates while late nominees have many, creating
    a mechanical correlation. Results are also stratified by decade.

    Returns dict with:
      - by_distance: DataFrame [distance, n_nominees, n_won, win_rate]
      - by_decade: dict {decade_str: DataFrame} with same structure
      - overall_reachable_rate: float
      - unreachable_rate: float
    """
    laureate_lookup = build_laureate_lookup(precomputed)

    # Build co-nomination graph
    G = build_conomination_graph(combined_df)

    # Identify laureate nodes and their win years
    laureate_nodes = {}  # node -> year_won
    for node in G.nodes:
        laureate = match_nominator_to_laureate(node, laureate_lookup)
        if laureate:
            laureate_nodes[node] = laureate["year_won"]

    # For each nominee, find their FIRST nomination year (more conservative
    # reference point — avoids inflating the set of "past laureates" for
    # nominees with long careers spanning many decades).
    nominee_first_year = {}
    nominee_won = {}
    for row in combined_df.itertuples(index=False):
        name = row.nominee_name
        yr = int(row.year)
        if name in G.nodes:
            if name not in nominee_first_year or yr < nominee_first_year[name]:
                nominee_first_year[name] = yr
            if hasattr(row, "nominee_prize_year") and pd.notna(row.nominee_prize_year):
                nominee_won[name] = True

    # Compute shortest-path distance to nearest past laureate for each non-laureate
    # Key structure: (distance_bucket, decade) -> {n, won}
    bucket_counts = defaultdict(lambda: {"n": 0, "won": 0})

    non_laureate_nodes = [n for n in G.nodes if n not in laureate_nodes]

    for node in non_laureate_nodes:
        first_yr = nominee_first_year.get(node, 9999)
        decade = f"{(first_yr // 10) * 10}s"

        # Find past laureates (won strictly before this nominee's first nomination)
        past_laureates = {ln for ln, yw in laureate_nodes.items() if yw < first_yr}
        if not past_laureates:
            bucket_counts[("unreachable", decade)]["n"] += 1
            bucket_counts[("unreachable", "all")]["n"] += 1
            if node in nominee_won:
                bucket_counts[("unreachable", decade)]["won"] += 1
                bucket_counts[("unreachable", "all")]["won"] += 1
            continue

        # BFS from node to find nearest past laureate
        try:
            min_dist = None
            lengths = nx.single_source_shortest_path_length(G, node, cutoff=5)
            for target, dist in lengths.items():
                if target in past_laureates:
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
        except nx.NetworkXError:
            min_dist = None

        if min_dist is None:
            bucket = "unreachable"
        elif min_dist <= 3:
            bucket = str(min_dist)
        else:
            bucket = "4+"

        bucket_counts[(bucket, decade)]["n"] += 1
        bucket_counts[(bucket, "all")]["n"] += 1
        if node in nominee_won:
            bucket_counts[(bucket, decade)]["won"] += 1
            bucket_counts[(bucket, "all")]["won"] += 1

    # Build overall results table
    dist_labels = ["1", "2", "3", "4+", "unreachable"]

    def _build_table(period):
        rows = []
        for d in dist_labels:
            data = bucket_counts[(d, period)]
            n = data["n"]
            won = data["won"]
            rows.append({
                "distance": d,
                "n_nominees": n,
                "n_won": won,
                "win_rate": round(won / n, 4) if n > 0 else 0,
            })
        return pd.DataFrame(rows)

    by_distance = _build_table("all")

    # Per-decade tables
    decades = sorted({d for (_, d) in bucket_counts if d != "all"})
    by_decade = {}
    for dec in decades:
        dec_table = _build_table(dec)
        if dec_table["n_nominees"].sum() > 0:
            by_decade[dec] = dec_table

    reachable = by_distance[by_distance["distance"] != "unreachable"]
    unreachable = by_distance[by_distance["distance"] == "unreachable"]

    reachable_total = reachable["n_nominees"].sum()
    reachable_won = reachable["n_won"].sum()
    overall_reachable_rate = reachable_won / reachable_total if reachable_total > 0 else 0

    unreachable_total = unreachable["n_nominees"].sum()
    unreachable_won = unreachable["n_won"].sum()
    unreachable_rate_val = unreachable_won / unreachable_total if unreachable_total > 0 else 0

    return {
        "by_distance": by_distance,
        "by_decade": by_decade,
        "overall_reachable_rate": round(overall_reachable_rate, 4),
        "unreachable_rate": round(unreachable_rate_val, 4),
    }


# ---------------------------------------------------------------------------
# ANALYSIS 6: NEAR-MISS ANALYSIS
# ---------------------------------------------------------------------------

def compute_near_miss_analysis(combined_df: pd.DataFrame, precomputed: dict,
                               min_nominations: int = 10) -> dict:
    """
    Do winners and 'near misses' (heavily nominated but never won) have
    structurally different nomination neighborhoods?

    For each person, computes:
      - Support breadth: number of unique nominators
      - Nominator reach: mean degree of their nominators in the nomination graph
      - Concentration: Herfindahl index of nominator contributions
      - Nominator diversity: number of unique communities among nominators

    Returns dict with near_miss_table, winner_table, comparison stats, top_near_misses.
    """
    from scipy.stats import mannwhitneyu

    laureate_lookup = build_laureate_lookup(precomputed)

    # Aggregate per nominee: total nominations, unique nominators, won?
    nominee_stats = {}
    nominator_counts_per_nominee = defaultdict(lambda: defaultdict(int))

    has_prize_col = "nominee_prize_year" in combined_df.columns
    for row in combined_df.itertuples(index=False):
        nominee = row.nominee_name
        nominator = row.nominator_name
        nominator_counts_per_nominee[nominee][nominator] += 1

        if nominee not in nominee_stats:
            won = has_prize_col and pd.notna(row.nominee_prize_year)
            nominee_stats[nominee] = {"won": bool(won), "total_noms": 0}
        nominee_stats[nominee]["total_noms"] += 1

    # Build nomination graph for computing nominator degrees
    G = build_nomination_graph(combined_df)

    # Build nominator co-activity graph: two nominators are linked if they
    # nominated any of the same people. This is the correct space for measuring
    # how many independent communities support a given nominee.
    nominator_to_nominees = defaultdict(set)
    for row in combined_df.itertuples(index=False):
        nominator_to_nominees[row.nominator_name].add(row.nominee_name)

    G_nom = nx.Graph()
    # Invert: for each nominee, link all pairs of their nominators
    nominee_to_nominators = defaultdict(set)
    for nominator, nominees in nominator_to_nominees.items():
        for nom in nominees:
            nominee_to_nominators[nom].add(nominator)

    nom_coact_edges = defaultdict(int)
    for nom, nominators in nominee_to_nominators.items():
        nominators = sorted(nominators)
        for i in range(len(nominators)):
            for j in range(i + 1, len(nominators)):
                nom_coact_edges[(nominators[i], nominators[j])] += 1
    for (n1, n2), w in nom_coact_edges.items():
        G_nom.add_edge(n1, n2, weight=w)

    # Identify near-misses and comparable winners
    near_misses = []
    winners = []
    for name, info in nominee_stats.items():
        if info["total_noms"] >= min_nominations:
            if not info["won"]:
                near_misses.append(name)
            else:
                winners.append(name)

    def compute_metrics(person_list):
        """Compute network metrics for a list of nominees."""
        rows = []
        for name in person_list:
            nom_counts = nominator_counts_per_nominee[name]
            total_noms = sum(nom_counts.values())
            breadth = len(nom_counts)

            # Nominator reach: mean degree of nominators in the full graph
            nominator_degrees = []
            for nominator in nom_counts:
                if nominator in G:
                    nominator_degrees.append(G.degree(nominator, weight="weight"))
            reach = np.mean(nominator_degrees) if nominator_degrees else 0

            # Herfindahl index: sum of squared shares
            if total_noms > 0:
                shares = [c / total_noms for c in nom_counts.values()]
                concentration = sum(s ** 2 for s in shares)
            else:
                concentration = 1.0

            # Nominator diversity: Louvain communities among this person's
            # nominators in the nominator co-activity graph (nominators linked
            # if they nominated any of the same people).
            nominator_set = set(nom_counts.keys())
            nominator_subgraph_nodes = [n for n in nominator_set if n in G_nom]
            if len(nominator_subgraph_nodes) > 2:
                sub = G_nom.subgraph(nominator_subgraph_nodes)
                if sub.number_of_edges() > 0:
                    try:
                        from networkx.algorithms.community import louvain_communities
                        communities = louvain_communities(sub, seed=42)
                        diversity = len(communities)
                    except Exception:
                        diversity = len(nominator_subgraph_nodes)
                else:
                    diversity = len(nominator_subgraph_nodes)
            else:
                diversity = max(len(nominator_subgraph_nodes), len(nominator_set))

            rows.append({
                "name": name,
                "total_noms": total_noms,
                "breadth": breadth,
                "diversity": diversity,
                "reach": round(reach, 2),
                "concentration": round(concentration, 4),
            })
        return pd.DataFrame(rows)

    near_miss_table = compute_metrics(near_misses)
    winner_table = compute_metrics(winners)

    # Statistical comparison: Mann-Whitney U for each metric
    comparison = {}
    for metric in ["breadth", "diversity", "reach", "concentration"]:
        w_vals = winner_table[metric].values if len(winner_table) > 0 else np.array([])
        nm_vals = near_miss_table[metric].values if len(near_miss_table) > 0 else np.array([])

        if len(w_vals) > 0 and len(nm_vals) > 0:
            u_stat, p_val = mannwhitneyu(w_vals, nm_vals, alternative="two-sided")
            comparison[metric] = {
                "winner_mean": round(float(np.mean(w_vals)), 3),
                "near_miss_mean": round(float(np.mean(nm_vals)), 3),
                "u_stat": round(float(u_stat), 1),
                "p_value": float(p_val),
            }
        else:
            comparison[metric] = {
                "winner_mean": 0, "near_miss_mean": 0,
                "u_stat": 0, "p_value": 1.0,
            }

    # Top near-misses by nomination count
    top_near_misses = near_miss_table.sort_values("total_noms", ascending=False).head(20)

    return {
        "near_miss_table": near_miss_table,
        "winner_table": winner_table,
        "comparison": comparison,
        "top_near_misses": top_near_misses,
        "n_near_misses": len(near_misses),
        "n_winners": len(winners),
    }


# ---------------------------------------------------------------------------
# ANALYSIS 7: CENTRALITY PREDICTS WINNERS
# ---------------------------------------------------------------------------

def compute_centrality_prediction(combined_df: pd.DataFrame, precomputed: dict) -> dict:
    """
    Does network position at time t predict winning at t+k?

    Builds cumulative directed nomination graphs at 5-year snapshots, computes
    centrality metrics for each nominee, and runs logistic regression to compare
    AUC-ROC of in-degree-only vs full-centrality models.

    CV uses leave-one-snapshot-out to respect temporal structure and avoid
    leaking information across correlated observations of the same nominee.

    Returns dict with AUC scores, coefficients, feature table, and ROC data.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve, auc

    laureate_lookup = build_laureate_lookup(precomputed)

    df = combined_df.copy()
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    snapshot_years = list(range(1910, 1970, 5))
    all_observations = []

    # Build cumulative graph incrementally — track the frontier year to avoid
    # re-processing edges that were already added at earlier snapshots.
    G = nx.DiGraph()
    edges_by_year = df.groupby("year")
    all_years = sorted(df["year"].unique())
    frontier = 0  # index into all_years: everything before this is already in G

    for snap_year in snapshot_years:
        # Add edges from frontier up to snap_year
        while frontier < len(all_years) and all_years[frontier] <= snap_year:
            yr = all_years[frontier]
            if yr in edges_by_year.groups:
                year_df = edges_by_year.get_group(yr)
                for row in year_df.itertuples(index=False):
                    n1, n2 = row.nominator_name, row.nominee_name
                    if G.has_edge(n1, n2):
                        G[n1][n2]["weight"] += 1
                    else:
                        G.add_edge(n1, n2, weight=1)
            frontier += 1

        if G.number_of_nodes() < 10:
            continue

        # Compute centrality measures for nominees (nodes with in-degree > 0)
        nominees_in_graph = [n for n in G.nodes if G.in_degree(n) > 0]
        if len(nominees_in_graph) < 10:
            continue

        # Weighted in-degree
        in_degrees = dict(G.in_degree(weight="weight"))

        # PageRank (weighted)
        try:
            pagerank = nx.pagerank(G, weight="weight", max_iter=100)
        except nx.PowerIterationFailedConvergence:
            pagerank = {n: 1.0 / G.number_of_nodes() for n in G.nodes}

        # Betweenness centrality (approximate)
        k_sample = min(200, G.number_of_nodes())
        betweenness = nx.betweenness_centrality(G, k=k_sample, weight="weight", seed=42)

        # Eigenvector centrality (on undirected version for convergence)
        G_undirected = G.to_undirected()
        try:
            eigenvector = nx.eigenvector_centrality(G_undirected, weight="weight", max_iter=200)
        except nx.PowerIterationFailedConvergence:
            eigenvector = {n: 0.0 for n in G.nodes}

        for nominee in nominees_in_graph:
            # Label: won within 10 years after snapshot?
            won = 0
            laureate = match_nominator_to_laureate(nominee, laureate_lookup)
            if laureate and snap_year < laureate["year_won"] <= snap_year + 10:
                won = 1

            all_observations.append({
                "snapshot": snap_year,
                "nominee": nominee,
                "in_degree": in_degrees.get(nominee, 0),
                "pagerank": pagerank.get(nominee, 0),
                "betweenness": betweenness.get(nominee, 0),
                "eigenvector": eigenvector.get(nominee, 0),
                "won": won,
            })

    if not all_observations:
        return {"error": "Not enough data for centrality prediction"}

    feature_table = pd.DataFrame(all_observations)
    feature_names = ["in_degree", "pagerank", "betweenness", "eigenvector"]

    # --- Leave-one-snapshot-out CV ---
    # Each fold holds out one snapshot for testing and trains on the rest.
    # This respects temporal structure and avoids the same nominee appearing
    # in both train and test at the same snapshot.
    snapshots_present = sorted(feature_table["snapshot"].unique())

    if len(snapshots_present) < 3:
        return {"error": f"Only {len(snapshots_present)} snapshots with data — need at least 3 for CV"}

    y_true_all = []
    y_score_baseline_all = []
    y_score_full_all = []
    auc_baseline_scores = []
    auc_full_scores = []

    for held_out in snapshots_present:
        train_mask = feature_table["snapshot"] != held_out
        test_mask = feature_table["snapshot"] == held_out

        y_train = feature_table.loc[train_mask, "won"].values
        y_test = feature_table.loc[test_mask, "won"].values

        # Skip folds with no positive or no negative examples in test
        if y_test.sum() == 0 or (y_test == 0).sum() == 0:
            continue
        if y_train.sum() == 0 or (y_train == 0).sum() == 0:
            continue

        # Standardize on train, transform test
        scaler_b = StandardScaler()
        scaler_f = StandardScaler()

        X_train_b = scaler_b.fit_transform(feature_table.loc[train_mask, ["in_degree"]].values)
        X_test_b = scaler_b.transform(feature_table.loc[test_mask, ["in_degree"]].values)
        X_train_f = scaler_f.fit_transform(feature_table.loc[train_mask, feature_names].values)
        X_test_f = scaler_f.transform(feature_table.loc[test_mask, feature_names].values)

        # Model A: in-degree only
        model_a = LogisticRegression(max_iter=1000, random_state=42)
        model_a.fit(X_train_b, y_train)
        proba_a = model_a.predict_proba(X_test_b)[:, 1]

        # Model B: all centrality features
        model_b = LogisticRegression(max_iter=1000, random_state=42)
        model_b.fit(X_train_f, y_train)
        proba_b = model_b.predict_proba(X_test_f)[:, 1]

        fpr_a, tpr_a, _ = roc_curve(y_test, proba_a)
        fpr_b, tpr_b, _ = roc_curve(y_test, proba_b)
        auc_a = auc(fpr_a, tpr_a)
        auc_b = auc(fpr_b, tpr_b)

        if not (np.isnan(auc_a) or np.isnan(auc_b)):
            auc_baseline_scores.append(auc_a)
            auc_full_scores.append(auc_b)

        y_true_all.extend(y_test)
        y_score_baseline_all.extend(proba_a)
        y_score_full_all.extend(proba_b)

    if not auc_baseline_scores:
        return {"error": "No valid CV folds (each snapshot needs both winners and non-winners)"}

    # Final ROC curve from pooled out-of-fold predictions
    fpr_base, tpr_base, _ = roc_curve(y_true_all, y_score_baseline_all)
    fpr_full, tpr_full, _ = roc_curve(y_true_all, y_score_full_all)

    # Fit final model on all data for coefficients
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(feature_table[feature_names].values)
    y_all = feature_table["won"].values
    final_model = LogisticRegression(max_iter=1000, random_state=42)
    final_model.fit(X_all, y_all)
    coefficients = dict(zip(feature_names, final_model.coef_[0].tolist()))

    auc_baseline = float(np.mean(auc_baseline_scores))
    auc_full = float(np.mean(auc_full_scores))

    return {
        "auc_baseline": round(auc_baseline, 4),
        "auc_full": round(auc_full, 4),
        "auc_improvement": round(auc_full - auc_baseline, 4),
        "coefficients": {k: round(v, 4) for k, v in coefficients.items()},
        "feature_table": feature_table,
        "roc_data": {
            "fpr_baseline": fpr_base.tolist(),
            "tpr_baseline": tpr_base.tolist(),
            "fpr_full": fpr_full.tolist(),
            "tpr_full": tpr_full.tolist(),
        },
        "n_folds_used": len(auc_baseline_scores),
        "n_snapshots": len(snapshots_present),
    }


# ---------------------------------------------------------------------------
# RENDERING HELPER: ADVANCED ANALYSES
# ---------------------------------------------------------------------------

def _render_advanced_analyses(combined_df, precomputed, flags):
    """
    Render all advanced analysis sections.

    flags: dict with keys like run_centrality, run_near_miss, run_temporal,
           run_proximity, near_miss_min, show_centrality, show_near_miss,
           show_temporal, show_proximity.
    """
    import matplotlib.pyplot as plt

    # --- Temporal Evolution ---
    if flags.get("show_temporal"):
        st.subheader("Temporal Network Evolution")
        st.markdown(
            "**What this measures:** We build a cumulative nomination network for each "
            "year from 1901 to 1970, adding that year's edges to all prior edges. At each "
            "year we track four structural properties: **(a)** the fraction of the network "
            "in the giant connected component (GCC), **(b)** mean degree (average number "
            "of connections per person), **(c)** clustering coefficient (how often a "
            "person's contacts are also connected to each other), and **(d)** total "
            "network size.\n\n"
            "**What to expect:** The GCC fraction should rise sharply in the early decades "
            "as the network 'percolates' -- isolated clusters merge into one large "
            "component. Look for a phase transition around 1920-1930 (the dashed line "
            "marks 1925, a structural transition identified by Hansson & Schlich). Mean "
            "degree should grow steadily as repeat nominations accumulate. Per-category "
            "lines may diverge: Physics and Chemistry typically grow faster than "
            "Literature or Peace.\n\n"
            "**Interpretation:** A rapid rise in GCC fraction indicates the nomination "
            "community became *structurally integrated* -- most nominators and nominees "
            "are connected through some chain of shared nominations. This matters because "
            "information (and reputation) can flow through connected networks. A high "
            "clustering coefficient means nominations are locally clustered: the people "
            "who nominate the same candidate also tend to nominate each other's candidates."
        )
        if flags.get("run_temporal"):
            with st.spinner("Computing temporal evolution (building 70 cumulative graphs)..."):
                result = compute_temporal_evolution(combined_df)

            if "error" in result:
                st.error(result["error"])
            else:
                overall = result["overall"]
                by_category = result["by_category"]
                categories = result["category_list"]

                cat_colors = {
                    "Physics": "#1f77b4",
                    "Chemistry": "#2ca02c",
                    "Physiology or Medicine": "#d62728",
                    "Medicine": "#d62728",
                    "Literature": "#9467bd",
                    "Peace": "#ff7f0e",
                }

                fig, axes = plt.subplots(2, 2, figsize=(12, 9))

                # (a) GCC fraction
                ax = axes[0, 0]
                ax.plot(overall.index, overall["gcc_frac"], color="black",
                        linewidth=2, label="Overall")
                for cat in categories:
                    if cat in by_category:
                        cat_df = by_category[cat]
                        ax.plot(cat_df.index, cat_df["gcc_frac"],
                                color=cat_colors.get(cat, "#999"),
                                linewidth=1, alpha=0.7, label=cat)
                ax.axvline(x=1925, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylabel("GCC fraction")
                ax.set_title("(a) Giant Connected Component")
                ax.legend(fontsize=7, loc="lower right")

                # (b) Mean degree
                ax = axes[0, 1]
                ax.plot(overall.index, overall["mean_degree"], color="black",
                        linewidth=2, label="Overall")
                for cat in categories:
                    if cat in by_category:
                        cat_df = by_category[cat]
                        ax.plot(cat_df.index, cat_df["mean_degree"],
                                color=cat_colors.get(cat, "#999"),
                                linewidth=1, alpha=0.7, label=cat)
                ax.axvline(x=1925, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylabel("Mean degree")
                ax.set_title("(b) Mean Degree")
                ax.legend(fontsize=7, loc="upper left")

                # (c) Clustering coefficient
                ax = axes[1, 0]
                ax.plot(overall.index, overall["clustering"], color="black",
                        linewidth=2)
                ax.axvline(x=1925, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylabel("Clustering coefficient")
                ax.set_xlabel("Year")
                ax.set_title("(c) Clustering Coefficient")

                # (d) Number of nodes
                ax = axes[1, 1]
                ax.plot(overall.index, overall["nodes"], color="black",
                        linewidth=2, label="Overall")
                for cat in categories:
                    if cat in by_category:
                        cat_df = by_category[cat]
                        ax.plot(cat_df.index, cat_df["nodes"],
                                color=cat_colors.get(cat, "#999"),
                                linewidth=1, alpha=0.7, label=cat)
                ax.axvline(x=1925, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylabel("Number of nodes")
                ax.set_xlabel("Year")
                ax.set_title("(d) Network Size")
                ax.legend(fontsize=7, loc="upper left")

                fig.tight_layout()
                st.pyplot(fig)
                _fig_download_buttons(fig, "temporal_evolution", "temporal")
                plt.close(fig)

                st.caption(
                    "Vertical dashed line marks 1925 (Hansson & Schlich transition point). "
                    "Note: Physiology/Medicine data ends at 1953."
                )

                # Download CSV
                _csv_download_button(overall.reset_index(), "temporal_evolution.csv",
                                     key="temporal_csv")
        else:
            st.info("Click **Run Evolution Analysis** in the sidebar.")

    # --- Three Degrees of Influence ---
    if flags.get("show_proximity"):
        st.subheader("Three Degrees of Influence")
        st.markdown(
            "**What this measures:** In the co-nomination network (where nominees are "
            "linked when they share a nominator), we compute each non-laureate nominee's "
            "shortest-path distance to the nearest *past* laureate -- someone who won "
            "strictly before that nominee's last nomination year. We then group nominees "
            "by distance (1 hop, 2 hops, 3 hops, 4+, or unreachable) and compute the "
            "win rate in each bucket.\n\n"
            "**What to expect:** If the 'three degrees of influence' hypothesis holds, "
            "win rates should decrease with distance: nominees who are 1 hop from a past "
            "laureate should win more often than those 2 or 3 hops away. Unreachable "
            "nominees (in a different connected component) should have the lowest win rate. "
            "The effect may be strongest within 1-2 hops and flatten beyond 3.\n\n"
            "**Interpretation:** A declining win rate with distance suggests that being "
            "embedded in the same nomination neighborhood as past laureates is predictive "
            "of success. This could reflect genuine influence (laureates advocate for "
            "nearby nominees), shared quality (similar-caliber scientists are co-nominated "
            "by the same experts), or institutional clustering (elite departments produce "
            "both laureates and future laureates). The analysis cannot distinguish these "
            "mechanisms, but a strong distance gradient is evidence that network position "
            "matters."
        )
        if flags.get("run_proximity"):
            with st.spinner("Computing proximity to laureates (BFS from each nominee)..."):
                result = compute_proximity_effect(combined_df, precomputed)

            if "error" in result:
                st.error(result["error"])
            else:
                by_dist = result["by_distance"]

                col1, col2, col3 = st.columns(3)
                dist1_rate = by_dist.loc[by_dist["distance"] == "1", "win_rate"]
                dist3p_rate = by_dist.loc[by_dist["distance"].isin(["3", "4+"]), "win_rate"]
                col1.metric("Win rate (distance 1)",
                            f"{float(dist1_rate.iloc[0]) * 100:.1f}%" if len(dist1_rate) > 0 else "N/A")
                col2.metric("Win rate (distance 3+)",
                            f"{float(dist3p_rate.mean()) * 100:.1f}%" if len(dist3p_rate) > 0 else "N/A")
                col3.metric("Win rate (unreachable)",
                            f"{result['unreachable_rate'] * 100:.1f}%")

                fig, ax = plt.subplots(figsize=(7, 4))
                x_labels = by_dist["distance"].values
                win_rates = by_dist["win_rate"].values * 100
                colors = ["#2ca02c", "#98df8a", "#ffbb78", "#ff7f0e", "#d62728"]
                bars = ax.bar(x_labels, win_rates, color=colors[:len(x_labels)],
                              edgecolor="black")
                ax.set_xlabel("Distance to nearest past laureate")
                ax.set_ylabel("Win rate (%)")
                ax.set_title("Proximity to Past Laureates and Win Rate")
                max_rate = max(win_rates) if max(win_rates) > 0 else 10
                ax.set_ylim(0, max_rate * 1.55)
                for bar, rate, row_data in zip(bars, win_rates, by_dist.itertuples()):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max_rate * 0.03,
                            f"{rate:.1f}% (n={row_data.n_nominees})",
                            ha="center", va="bottom", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                _fig_download_buttons(fig, "proximity_effect", "proximity")
                plt.close(fig)

                st.caption(
                    "Distance measured in the co-nomination network (nominees sharing "
                    "nominators). 'Past laureate' = won strictly before the nominee's "
                    "first nomination year."
                )
                _csv_download_button(by_dist, "proximity_distance.csv",
                                     key="proximity_csv")

                # Decade stratification
                by_decade = result.get("by_decade", {})
                if by_decade:
                    st.markdown("#### Win Rate by Distance, Stratified by Decade")
                    st.caption(
                        "Controls for time-period bias: early nominees have few "
                        "past laureates to be near, inflating distances."
                    )
                    decade_rows = []
                    for dec in sorted(by_decade.keys()):
                        dec_df = by_decade[dec]
                        for _, r in dec_df.iterrows():
                            decade_rows.append({
                                "Decade": dec,
                                "Distance": r["distance"],
                                "N": r["n_nominees"],
                                "Won": r["n_won"],
                                "Win rate": f"{r['win_rate'] * 100:.1f}%",
                            })
                    st.dataframe(pd.DataFrame(decade_rows), hide_index=True)
        else:
            st.info("Click **Run Proximity Analysis** in the sidebar.")

    # --- Near-Miss Analysis ---
    if flags.get("show_near_miss"):
        st.subheader("Near-Miss Analysis")
        st.markdown(
            "**What this measures:** We identify 'near-misses' -- nominees who received "
            "many nominations but never won -- and compare their nomination neighborhoods "
            "to winners with similar nomination counts. Four metrics capture different "
            "aspects of support structure:\n"
            "- **Support breadth**: number of unique nominators (wide vs. narrow support)\n"
            "- **Nominator diversity**: number of distinct communities among a nominee's "
            "nominators (are they championed by one clique or multiple independent groups?)\n"
            "- **Nominator reach**: mean degree of a nominee's nominators (are they "
            "nominated by well-connected, influential people or by peripheral ones?)\n"
            "- **Concentration** (Herfindahl index): how evenly spread the nominations "
            "are across nominators (1.0 = all from one person; low = distributed)\n\n"
            "**What to expect:** If winning depends on more than raw nomination count, "
            "winners should differ from near-misses on these structural metrics. "
            "Specifically, winners may have *broader* support (more unique nominators), "
            "*more diverse* nominator communities, and *higher-reach* nominators. "
            "Near-misses may show higher concentration (intense support from a small "
            "group that wasn't enough).\n\n"
            "**Interpretation:** The box plots compare distributions and the Mann-Whitney "
            "U test reports whether the difference is statistically significant "
            "(p < 0.05). A significant difference in breadth or diversity suggests that "
            "*who* supports you and *how many independent groups* back you matters more "
            "than raw nomination volume alone. This aligns with the idea that the Nobel "
            "committee is more persuaded by broad, independent consensus than by a "
            "concentrated campaign from a single community."
        )
        if flags.get("run_near_miss"):
            min_noms = flags.get("near_miss_min", 10)
            with st.spinner(f"Analyzing near-misses (min {min_noms} nominations)..."):
                result = compute_near_miss_analysis(combined_df, precomputed,
                                                    min_nominations=min_noms)

            col1, col2 = st.columns(2)
            col1.metric("Near-misses", result["n_near_misses"])
            col2.metric("Comparable winners", result["n_winners"])

            # Box plots: 2x2 grid
            metrics = ["breadth", "diversity", "reach", "concentration"]
            titles = ["Support Breadth\n(unique nominators)",
                      "Nominator Diversity\n(communities)",
                      "Nominator Reach\n(mean nominator degree)",
                      "Concentration\n(Herfindahl index)"]

            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[i // 2, i % 2]
                w_vals = result["winner_table"][metric].values if len(result["winner_table"]) > 0 else []
                nm_vals = result["near_miss_table"][metric].values if len(result["near_miss_table"]) > 0 else []

                data = []
                labels = []
                if len(w_vals) > 0:
                    data.append(w_vals)
                    labels.append("Winners")
                if len(nm_vals) > 0:
                    data.append(nm_vals)
                    labels.append("Near-misses")

                if data:
                    bp = ax.boxplot(data, labels=labels, patch_artist=True)
                    colors_box = ["#FFD700", "#999999"]
                    for patch, color in zip(bp["boxes"], colors_box[:len(data)]):
                        patch.set_facecolor(color)

                comp = result["comparison"].get(metric, {})
                p_val = comp.get("p_value", 1.0)
                ax.set_title(f"{title}\n(p = {p_val:.2e})", fontsize=10)

            fig.tight_layout()
            st.pyplot(fig)
            _fig_download_buttons(fig, "near_miss_comparison", "near_miss_plot")
            plt.close(fig)

            # Comparison table
            st.markdown("#### Statistical Comparison (Mann-Whitney U)")
            comp_rows = []
            for metric in metrics:
                comp = result["comparison"][metric]
                comp_rows.append({
                    "Metric": metric.replace("_", " ").title(),
                    "Winner Mean": comp["winner_mean"],
                    "Near-Miss Mean": comp["near_miss_mean"],
                    "U statistic": comp["u_stat"],
                    "p-value": f"{comp['p_value']:.2e}",
                })
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True)

            # Top near-misses
            st.markdown("#### Top 20 Near-Misses by Nomination Count")
            st.dataframe(result["top_near_misses"], hide_index=True)
            _csv_download_button(result["near_miss_table"], "near_misses.csv",
                                 key="near_miss_csv")
            _csv_download_button(result["winner_table"], "comparable_winners.csv",
                                 key="winners_csv")
        else:
            st.info("Click **Run Near-Miss Analysis** in the sidebar.")

    # --- Centrality Predicts Winners ---
    if flags.get("show_centrality"):
        st.subheader("Centrality Predicts Winners")
        st.markdown(
            "**What this measures:** At 5-year snapshots (1910, 1915, ..., 1965), we "
            "build the cumulative nomination network and compute four centrality measures "
            "for each nominee:\n"
            "- **In-degree** (weighted): raw nomination count -- the baseline\n"
            "- **PageRank**: a nominee's importance weighted by the importance of their "
            "nominators (being nominated by someone who is themselves highly nominated "
            "counts more)\n"
            "- **Betweenness centrality**: how often a nominee lies on the shortest path "
            "between other people in the network (bridge positions)\n"
            "- **Eigenvector centrality**: similar to PageRank but for undirected "
            "influence -- being connected to other well-connected people\n\n"
            "We then ask: does a nominee win within 10 years after the snapshot? Two "
            "logistic regression models are compared using 5-fold cross-validated "
            "AUC-ROC: **(A)** in-degree only (does raw nomination count predict winning?) "
            "vs. **(B)** all four centrality features (does structural position add "
            "predictive power?).\n\n"
            "**What to expect:** The baseline AUC should be above 0.5 (nomination count "
            "alone has some predictive value). If structural position matters, the full "
            "model's AUC should be meaningfully higher. The coefficient table reveals "
            "which centrality feature contributes most.\n\n"
            "**Interpretation:** An AUC improvement from adding PageRank/betweenness/"
            "eigenvector means that *who* nominates you carries information beyond *how "
            "many* people nominate you. A large positive PageRank coefficient, for "
            "example, would mean that being nominated by influential nominators predicts "
            "winning. The ROC curve visualizes the tradeoff between true positives "
            "(correctly predicted winners) and false positives (non-winners predicted to "
            "win) -- a curve closer to the top-left corner indicates better prediction."
        )
        if flags.get("run_centrality"):
            with st.spinner("Running centrality analysis (PageRank, betweenness, logistic regression)..."):
                result = compute_centrality_prediction(combined_df, precomputed)

            if "error" in result:
                st.error(result["error"])
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("AUC (in-degree only)", f"{result['auc_baseline']:.3f}")
                col2.metric("AUC (all centrality)", f"{result['auc_full']:.3f}")
                col3.metric("Improvement", f"+{result['auc_improvement']:.3f}")

                # ROC curve
                roc = result["roc_data"]
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(roc["fpr_baseline"], roc["tpr_baseline"],
                        color="#999999", linewidth=2,
                        label=f"In-degree only (AUC={result['auc_baseline']:.3f})")
                ax.plot(roc["fpr_full"], roc["tpr_full"],
                        color="#d62728", linewidth=2,
                        label=f"All centrality (AUC={result['auc_full']:.3f})")
                ax.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.3)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve: Predicting Nobel Winners from Network Position")
                ax.legend(loc="lower right")
                fig.tight_layout()
                st.pyplot(fig)
                _fig_download_buttons(fig, "centrality_roc", "centrality_roc")
                plt.close(fig)

                # Coefficient table
                st.markdown("#### Feature Coefficients (Logistic Regression)")
                coef_df = pd.DataFrame([
                    {"Feature": k, "Coefficient": v}
                    for k, v in sorted(result["coefficients"].items(),
                                       key=lambda x: abs(x[1]), reverse=True)
                ])
                st.dataframe(coef_df, hide_index=True)

                st.caption(
                    "Positive coefficients indicate features that increase the "
                    "probability of winning. Coefficients are on standardized features."
                )

                _csv_download_button(result["feature_table"],
                                     "centrality_features.csv",
                                     key="centrality_csv")
        else:
            st.info("Click **Run Centrality Analysis** in the sidebar.")
