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

def render_network_page(df: pd.DataFrame, precomputed: dict | None = None,
                        combined_df: pd.DataFrame | None = None):
    """
    Full Streamlit page for network analysis.
    Category and year range are already filtered by the caller (sidebar).
    """
    if not HAS_STREAMLIT:
        print("Streamlit not available. Use this module within a Streamlit app.")
        return

    has_combined = combined_df is not None and not combined_df.empty
    has_precomputed = precomputed is not None and len(precomputed) > 0

    # --- Sidebar filters (below the data selection controls) ---
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

    # Default min_weight for cross-category (too many nodes at 1)
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

    # --- Main area ---
    st.header("Nomination Networks")

    # Build the appropriate graph
    if network_type == "Nominator -> Nominee":
        G = build_nomination_graph(df, country=country_filter)
    elif network_type == "Co-nomination (shared nominators)":
        G = build_conomination_graph(df)
    elif is_cross_category:
        with st.spinner("Building cross-category combined network..."):
            G = build_combined_nomination_graph(combined_df, precomputed)

    # Render interactive visualization first
    if G.number_of_nodes() > 0:
        color_mode = "category" if is_cross_category else "country"
        html_path = visualize_graph(
            G, title=network_type, min_edge_weight=min_weight, color_by=color_mode,
        )
        if html_path:
            with open(html_path, "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=720, scrolling=True)
            os.unlink(html_path)
        else:
            st.info("No edges meet the current filter criteria.")
    else:
        st.info("No data to display for current filters.")

    # --- Everything below the figure ---
    if G.number_of_nodes() > 0:
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

    # --- Main area: render selected analyses ---

    if show_campaigns and run_campaigns:
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

    if show_paper:
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

        import matplotlib.pyplot as plt
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

    if show_raw:
        st.subheader("Raw Edge Data")
        st.dataframe(df, hide_index=True)
        _csv_download_button(df, "nomination_edges.csv", key="raw_edges_csv")


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

    for _, row in df.iterrows():
        nominee = row["nominee_name"]
        nom_year = row["year"]
        nominator = row["nominator_name"]

        if nominee not in nominee_info:
            won = pd.notna(row.get("nominee_prize_year")) if "nominee_prize_year" in row.index else False
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

    for _, row in df.iterrows():
        nominator = row["nominator_name"]
        nominee = row["nominee_name"]
        cat = row.get("category", "Unknown")

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

    Null model: randomly pick n_unique_laureates nodes from the graph,
    count how many land in the LCC, repeat 1000 times.

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

    # Null model: randomly pick n_laureates nodes, count how many land in LCC
    all_nodes = list(G.nodes)
    rng = random.Random(42)
    null_counts = []
    for _ in range(n_permutations):
        random_picks = set(rng.sample(all_nodes, n_laureates))
        null_counts.append(len(random_picks & lcc))

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
    }
