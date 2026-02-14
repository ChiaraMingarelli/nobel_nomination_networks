"""
Nobel Prize Nomination Network Analysis — Bayesian Edition
===========================================================
Three network views + Bayesian statistical analyses (PyMC + ArviZ).
Dependencies: pip install networkx pyvis pandas streamlit pymc arviz
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
    filtered = _filter(df, category, year_range, country)
    nominee_counts = filtered.groupby("nominee_name").size().to_dict()
    G = nx.DiGraph()
    for (nominator, nominee), group in filtered.groupby(["nominator_name", "nominee_name"]):
        weight = len(group)
        years = sorted(group["year"].unique())
        G.add_edge(nominator, nominee, weight=weight, years=years)
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
# 2. CO-NOMINATION NETWORK
# ---------------------------------------------------------------------------

def build_conomination_graph(df: pd.DataFrame,
                              category: str | None = None,
                              year_range: tuple[int, int] | None = None) -> nx.Graph:
    filtered = _filter(df, category, year_range)
    nominator_to_nominees = defaultdict(set)
    for row in filtered.itertuples(index=False):
        nominator_to_nominees[row.nominator_name].add(row.nominee_name)
    MAX_NOMINEES_PER_NOMINATOR = 50
    G = nx.Graph()
    edge_weights = defaultdict(int)
    for nominator, nominees in nominator_to_nominees.items():
        nominees = sorted(nominees)
        if len(nominees) > MAX_NOMINEES_PER_NOMINATOR:
            continue
        for i in range(len(nominees)):
            for j in range(i + 1, len(nominees)):
                edge_weights[(nominees[i], nominees[j])] += 1
    nominee_countries = {}
    nominee_prize_years = {}
    nominee_counts = filtered.groupby("nominee_name").size().to_dict()
    if "nominee_country" in filtered.columns:
        nominee_countries = (
            filtered.dropna(subset=["nominee_country"])
            .groupby("nominee_name")["nominee_country"].first().to_dict()
        )
    if "nominee_prize_year" in filtered.columns:
        prize_rows = filtered.dropna(subset=["nominee_prize_year"])
        if len(prize_rows) > 0:
            nominee_prize_years = (
                prize_rows.groupby("nominee_name")["nominee_prize_year"]
                .first().astype(int).to_dict()
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
    filtered = _filter(df, category, year_range)
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
    filtered = df.copy()
    campaigns = []
    for nominee, group in filtered.groupby("nominee_name"):
        group = group.sort_values("year")
        years = group["year"].values
        if len(years) == 0:
            continue
        raw_windows = []
        for start_year in range(int(years.min()), int(years.max()) - year_window + 2):
            end_year = start_year + year_window - 1
            window = group[(group["year"] >= start_year) & (group["year"] <= end_year)]
            if len(window) >= min_nominations:
                nominators = window["nominator_name"].unique()
                raw_windows.append({
                    "year_start": start_year, "year_end": end_year,
                    "n_nominations": len(window),
                    "n_unique_nominators": len(nominators),
                    "nominators": list(nominators),
                })
        if not raw_windows:
            continue
        bursts = []
        current_burst = [raw_windows[0]]
        for w in raw_windows[1:]:
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
                "year_start": best["year_start"], "year_end": best["year_end"],
                "n_nominations": best["n_nominations"],
                "n_unique_nominators": best["n_unique_nominators"],
                "nominators": best["nominators"],
            })
    if not campaigns:
        return pd.DataFrame(columns=["nominee", "year_start", "year_end",
                                     "n_nominations", "n_unique_nominators", "nominators"])
    return pd.DataFrame(campaigns).sort_values("n_nominations", ascending=False)


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

COUNTRY_COLORS = {
    "USA": "#1f77b4", "Canada": "#d62728", "Germany": "#2ca02c",
    "France": "#9467bd", "UK": "#ff7f0e", "Sweden": "#e377c2",
    "Italy": "#8c564b", "the Netherlands": "#17becf", "Switzerland": "#bcbd22",
    "Austria": "#7f7f7f", "Denmark": "#aec7e8", "Russia": "#ff9896",
    "Japan": "#98df8a", "Unknown": "#cccccc",
}


def visualize_graph(G, title="Nobel Nomination Network", height="700px",
                    width="100%", size_by="total_nominations", color_by="country",
                    min_edge_weight=1, physics=True):
    if min_edge_weight > 1:
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                           if d.get("weight", 1) < min_edge_weight]
        G = G.copy()
        G.remove_edges_from(edges_to_remove)
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)
    if len(G.nodes) == 0:
        return None
    directed = isinstance(G, nx.DiGraph)
    net = Network(height=height, width=width, directed=directed,
                  notebook=False, cdn_resources="remote")
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
        node_opts = {"label": label, "size": size, "color": color, "title": title_text}
        if data.get("is_laureate"):
            node_opts["borderWidth"] = 3
            node_opts["color"] = {"background": color, "border": "#FFD700"}
        net.add_node(node, **node_opts)
    for u, v, d in G.edges(data=True):
        weight = d.get("weight", 1)
        title_text = f"Weight: {weight}"
        if "years" in d:
            title_text += f"\nYears: {d['years']}"
        net.add_edge(u, v, value=weight, title=title_text)
    if physics:
        net.force_atlas_2based(gravity=-50, central_gravity=0.01,
                               spring_length=100, spring_strength=0.08)
    else:
        net.toggle_physics(False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w")
    net.save_graph(tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# CATEGORY COLORS
# ---------------------------------------------------------------------------

DISCIPLINE_COLORS = {
    "Physics": "#1f77b4", "Chemistry": "#2ca02c",
    "Physiology or Medicine": "#d62728", "Medicine": "#d62728",
    "Literature": "#9467bd", "Peace": "#ff7f0e", "nominator": "#999999",
}


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


def _fig_download_buttons(fig, filename_stem, key_prefix):
    col_pdf, col_png, _ = st.columns([1, 1, 3])
    buf_pdf = io.BytesIO()
    try:
        fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
        col_pdf.download_button("Download PDF", buf_pdf.getvalue(),
                                file_name=f"{filename_stem}.pdf", mime="application/pdf",
                                key=f"{key_prefix}_pdf")
    except ValueError:
        pass
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
    col_png.download_button("Download PNG", buf_png.getvalue(),
                            file_name=f"{filename_stem}.png", mime="image/png",
                            key=f"{key_prefix}_png")


def _csv_download_button(dataframe, filename, key, label="Download CSV"):
    csv_data = dataframe.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv_data, file_name=filename, mime="text/csv", key=key)


# ---------------------------------------------------------------------------
# NAME MATCHING
# ---------------------------------------------------------------------------

def normalize_laureate_name(name):
    name = name.lower().strip()
    for suffix in [", jr.", ", jr", " jr.", " jr", ", sr.", ", sr", " sr.", " sr"]:
        name = name.replace(suffix, "")
    tokens = name.split()
    filtered = []
    for t in tokens:
        if len(t) <= 2 and (len(t) == 1 or t.endswith(".")):
            continue
        filtered.append(t)
    return " ".join(filtered)


def build_laureate_lookup(precomputed):
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
            lookup[norm] = {"name": name, "year_won": int(year_won),
                            "category": category, "id": eid}
    return lookup


def match_nominator_to_laureate(nominator_name, laureate_lookup):
    norm = normalize_laureate_name(nominator_name)
    if norm in laureate_lookup:
        return laureate_lookup[norm]
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
# ANALYSIS 1: LAUREATE ENDORSEMENT EFFECT (unchanged)
# ---------------------------------------------------------------------------

def compute_endorsement_effect(df, precomputed):
    laureate_lookup = build_laureate_lookup(precomputed)
    nominee_info = {}
    has_prize_col = "nominee_prize_year" in df.columns
    for row in df.itertuples(index=False):
        nominee = row.nominee_name
        nom_year = row.year
        nominator = row.nominator_name
        if nominee not in nominee_info:
            won = has_prize_col and pd.notna(row.nominee_prize_year)
            nominee_info[nominee] = {"endorsed": False, "won": bool(won)}
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
        "endorsed_total": endorsed_total, "endorsed_won": endorsed_won,
        "endorsed_rate": endorsed_rate, "not_endorsed_total": not_endorsed_total,
        "not_endorsed_won": not_endorsed_won, "not_endorsed_rate": not_endorsed_rate,
        "ratio": ratio,
    }


# ---------------------------------------------------------------------------
# ANALYSIS 2: CROSS-CATEGORY COMBINED NETWORK (unchanged)
# ---------------------------------------------------------------------------

def build_combined_nomination_graph(df, precomputed):
    laureate_lookup = build_laureate_lookup(precomputed)
    G = nx.Graph()
    edge_weights = defaultdict(int)
    node_categories = defaultdict(lambda: defaultdict(int))
    for row in df.itertuples(index=False):
        nominator = row.nominator_name
        nominee = row.nominee_name
        cat = getattr(row, "category", "Unknown") or "Unknown"
        edge_weights[(nominator, nominee)] += 1
        node_categories[nominee][cat] += 1
        node_categories[nominator][cat] += 1
    for (n1, n2), weight in edge_weights.items():
        G.add_edge(n1, n2, weight=weight)
    for node in G.nodes:
        laureate = match_nominator_to_laureate(node, laureate_lookup)
        if laureate:
            G.nodes[node]["category"] = laureate["category"]
            G.nodes[node]["is_laureate"] = True
            G.nodes[node]["prize_year"] = laureate["year_won"]
        else:
            cats = node_categories.get(node, {})
            G.nodes[node]["category"] = max(cats, key=cats.get) if cats else "Unknown"
            G.nodes[node]["is_laureate"] = False
        G.nodes[node]["total_nominations"] = sum(
            d.get("weight", 1) for _, _, d in G.edges(node, data=True))
    return G


# ---------------------------------------------------------------------------
# ANALYSIS 3: LAUREATES IN LCC (unchanged)
# ---------------------------------------------------------------------------

def compute_lcc_analysis(G, precomputed, n_permutations=1000):
    laureate_lookup = build_laureate_lookup(precomputed)
    node_to_lid = {}
    for node in G.nodes:
        laureate = match_nominator_to_laureate(node, laureate_lookup)
        if laureate:
            node_to_lid[node] = laureate["id"]
    unique_lids = set(node_to_lid.values())
    n_laureates = len(unique_lids)
    if n_laureates == 0:
        return {"error": "No laureates found in graph"}
    components = list(nx.connected_components(G))
    if not components:
        return {"error": "No connected components"}
    lcc = max(components, key=len)
    lcc_size = len(lcc)
    graph_size = G.number_of_nodes()
    lids_in_lcc = {node_to_lid[n] for n in node_to_lid if n in lcc}
    observed = len(lids_in_lcc)
    nominee_nodes = []
    for node in G.nodes:
        data = G.nodes[node]
        if data.get("role") == "nominee" or data.get("is_laureate") or data.get("total_nominations", 0) > 0:
            nominee_nodes.append(node)
    if len(nominee_nodes) < n_laureates:
        nominee_nodes = list(G.nodes)
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
    p_value = math.erfc(z_score / math.sqrt(2)) / 2
    return {
        "observed": observed, "expected_mean": round(expected_mean, 1),
        "expected_std": round(expected_std, 2), "z_score": round(z_score, 2),
        "p_value": p_value, "lcc_size": lcc_size, "graph_size": graph_size,
        "n_laureates": n_laureates, "null_pool_size": len(nominee_nodes),
    }


# ---------------------------------------------------------------------------
# ANALYSIS 4: TEMPORAL EVOLUTION (unchanged)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self):
        self._parent = {}
        self._rank = {}
        self._size = {}
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


def compute_temporal_evolution(combined_df):
    df = combined_df.copy()
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    min_year = max(1901, df["year"].min())
    max_year = min(1970, df["year"].max())
    years = list(range(min_year, max_year + 1))
    categories = sorted(df["category"].dropna().unique())
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
            if n_nodes > 5000:
                sample_nodes = random.sample(list(G.nodes), min(2000, n_nodes))
                clustering = nx.average_clustering(G, nodes=sample_nodes)
            else:
                clustering = nx.average_clustering(G)
        else:
            gcc_size = gcc_frac = clustering = mean_degree = 0
            n_components = 0
        overall_rows.append({
            "year": yr, "nodes": n_nodes, "edges": n_edges,
            "gcc_size": gcc_size, "gcc_frac": round(gcc_frac, 4),
            "clustering": round(clustering, 4), "mean_degree": round(mean_degree, 3),
            "n_components": n_components,
        })
    overall = pd.DataFrame(overall_rows).set_index("year")
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
                gcc_size = 1; gcc_frac = 1.0; mean_degree = 0
            else:
                gcc_size = gcc_frac = mean_degree = 0
            cat_rows.append({
                "year": yr, "nodes": n_nodes, "edges": n_edges,
                "gcc_size": gcc_size, "gcc_frac": round(gcc_frac, 4),
                "mean_degree": round(mean_degree, 3),
            })
        by_category[cat] = pd.DataFrame(cat_rows).set_index("year")
    return {"overall": overall, "by_category": by_category, "category_list": categories}


# ---------------------------------------------------------------------------
# ANALYSIS 5: THREE DEGREES OF INFLUENCE (unchanged)
# ---------------------------------------------------------------------------

def compute_proximity_effect(combined_df, precomputed):
    laureate_lookup = build_laureate_lookup(precomputed)
    G = build_conomination_graph(combined_df)
    laureate_nodes = {}
    for node in G.nodes:
        laureate = match_nominator_to_laureate(node, laureate_lookup)
        if laureate:
            laureate_nodes[node] = laureate["year_won"]
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
    bucket_counts = defaultdict(lambda: {"n": 0, "won": 0})
    for node in G.nodes:
        first_yr = nominee_first_year.get(node, 9999)
        decade = f"{(first_yr // 10) * 10}s"
        past_laureates = {ln for ln, yw in laureate_nodes.items()
                          if yw < first_yr and ln != node}
        if not past_laureates:
            bucket_counts[("unreachable", decade)]["n"] += 1
            bucket_counts[("unreachable", "all")]["n"] += 1
            if node in nominee_won:
                bucket_counts[("unreachable", decade)]["won"] += 1
                bucket_counts[("unreachable", "all")]["won"] += 1
            continue
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
    dist_labels = ["1", "2", "3", "4+", "unreachable"]
    def _build_table(period):
        rows = []
        for d in dist_labels:
            data = bucket_counts[(d, period)]
            n = data["n"]
            won = data["won"]
            rows.append({"distance": d, "n_nominees": n, "n_won": won,
                          "win_rate": round(won / n, 4) if n > 0 else 0})
        return pd.DataFrame(rows)
    by_distance = _build_table("all")
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

    # Dist-1 vs Dist-2 Bayes Factor comparison
    from bayesian_models import conjugate_beta_binomial_bf
    proximity_bf = None
    dist1_row = by_distance[by_distance["distance"] == "1"]
    dist2_row = by_distance[by_distance["distance"] == "2"]
    if len(dist1_row) > 0 and len(dist2_row) > 0:
        d1_n = int(dist1_row.iloc[0]["n_nominees"])
        d1_won = int(dist1_row.iloc[0]["n_won"])
        d2_n = int(dist2_row.iloc[0]["n_nominees"])
        d2_won = int(dist2_row.iloc[0]["n_won"])
        if d1_n >= 2 and d2_n >= 2:
            proximity_bf = conjugate_beta_binomial_bf(
                d1_n, d1_won, d2_n, d2_won,
                ("distance_1", "distance_2"))

    return {
        "by_distance": by_distance, "by_decade": by_decade,
        "overall_reachable_rate": round(overall_reachable_rate, 4),
        "unreachable_rate": round(unreachable_rate_val, 4),
        "proximity_bf": proximity_bf,
    }


# ---------------------------------------------------------------------------
# ANALYSIS 6: NEAR-MISS — BAYESIAN (replaces Mann-Whitney U)
# ---------------------------------------------------------------------------

def compute_near_miss_analysis(combined_df, precomputed, min_nominations=10,
                                **sample_kwargs):
    from bayesian_models import bayesian_two_group_comparison

    laureate_lookup = build_laureate_lookup(precomputed)
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

    G = build_nomination_graph(combined_df)

    nominator_to_nominees = defaultdict(set)
    for row in combined_df.itertuples(index=False):
        nominator_to_nominees[row.nominator_name].add(row.nominee_name)
    nominee_to_nominators = defaultdict(set)
    for nominator, nominees in nominator_to_nominees.items():
        for nom in nominees:
            nominee_to_nominators[nom].add(nominator)
    G_nom = nx.Graph()
    nom_coact_edges = defaultdict(int)
    for nom, nominators in nominee_to_nominators.items():
        nominators = sorted(nominators)
        for i in range(len(nominators)):
            for j in range(i + 1, len(nominators)):
                nom_coact_edges[(nominators[i], nominators[j])] += 1
    for (n1, n2), w in nom_coact_edges.items():
        G_nom.add_edge(n1, n2, weight=w)

    near_misses = []
    winners = []
    for name, info in nominee_stats.items():
        if info["total_noms"] >= min_nominations:
            if not info["won"]:
                near_misses.append(name)
            else:
                winners.append(name)

    def compute_metrics(person_list):
        rows = []
        for name in person_list:
            nom_counts = nominator_counts_per_nominee[name]
            total_noms = sum(nom_counts.values())
            breadth = len(nom_counts)
            nominator_degrees = []
            for nominator in nom_counts:
                if nominator in G:
                    nominator_degrees.append(G.degree(nominator, weight="weight"))
            reach = np.mean(nominator_degrees) if nominator_degrees else 0
            if total_noms > 0:
                shares = [c / total_noms for c in nom_counts.values()]
                concentration = sum(s ** 2 for s in shares)
            else:
                concentration = 1.0
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
                "name": name, "total_noms": total_noms, "breadth": breadth,
                "diversity": diversity, "reach": round(reach, 2),
                "concentration": round(concentration, 4),
            })
        return pd.DataFrame(rows)

    near_miss_table = compute_metrics(near_misses)
    winner_table = compute_metrics(winners)

    # Bayesian comparison for each metric
    draws = sample_kwargs.get("draws", 1000)
    tune = sample_kwargs.get("tune", 500)
    chains = sample_kwargs.get("chains", 2)

    comparison = {}
    bayesian_results = {}
    for metric in ["breadth", "diversity", "reach", "concentration"]:
        w_vals = winner_table[metric].values if len(winner_table) > 0 else np.array([])
        nm_vals = near_miss_table[metric].values if len(near_miss_table) > 0 else np.array([])
        if len(w_vals) > 1 and len(nm_vals) > 1:
            idata, summary = bayesian_two_group_comparison(
                w_vals, nm_vals, metric, draws=draws, tune=tune, chains=chains)
            comparison[metric] = summary
            bayesian_results[metric] = idata
        else:
            comparison[metric] = {
                "diff_mean": 0, "hdi_low": 0, "hdi_high": 0,
                "p_greater": 0.5, "mu1_mean": 0, "mu2_mean": 0,
                "metric_name": metric,
            }

    top_near_misses = near_miss_table.sort_values("total_noms", ascending=False).head(20)

    return {
        "near_miss_table": near_miss_table, "winner_table": winner_table,
        "comparison": comparison, "bayesian_results": bayesian_results,
        "top_near_misses": top_near_misses,
        "n_near_misses": len(near_misses), "n_winners": len(winners),
    }


# ---------------------------------------------------------------------------
# ANALYSIS 7: CAMPAIGN SUCCESS — BAYESIAN (replaces Fisher + CMH)
# ---------------------------------------------------------------------------

def compute_campaign_success(combined_df, precomputed, min_nominations=5,
                              year_window=3, **sample_kwargs):
    from bayesian_models import (bayesian_beta_binomial, bayesian_hierarchical_logistic,
                                  conjugate_beta_binomial_bf)

    # --- Step 1: Relative burst detection (same as original) ---
    raw_campaigns_df = detect_campaigns(combined_df, min_nominations=min_nominations,
                                        year_window=year_window)
    if raw_campaigns_df.empty:
        return {"error": "No campaigns detected with current thresholds."}

    nominee_year_counts = combined_df.groupby(
        ["nominee_name", "year"]).size().reset_index(name="count")

    filtered_campaigns = []
    for _, row in raw_campaigns_df.iterrows():
        nominee = row["nominee"]
        nom_years = nominee_year_counts[nominee_year_counts["nominee_name"] == nominee]
        if nom_years.empty:
            continue
        window_years = row["year_end"] - row["year_start"] + 1
        window_annual_rate = row["n_nominations"] / window_years
        burst_year_set = set(range(int(row["year_start"]), int(row["year_end"]) + 1))
        baseline = nom_years[~nom_years["year"].isin(burst_year_set)]["count"]
        if len(baseline) <= 1:
            filtered_campaigns.append(row)
        else:
            base_mean = float(baseline.mean())
            base_std = float(baseline.std())
            threshold = max(base_mean + 2 * base_std, base_mean * 1.5)
            if window_annual_rate > threshold:
                filtered_campaigns.append(row)

    if not filtered_campaigns:
        return {"error": "No anomalous campaigns detected."}

    campaigns_df = pd.DataFrame(filtered_campaigns)
    campaign_names = set(campaigns_df["nominee"].unique())

    # --- Step 2: Build per-nominee stats ---
    has_prize_col = "nominee_prize_year" in combined_df.columns
    nominee_stats = {}
    for row in combined_df.itertuples(index=False):
        nominee = row.nominee_name
        if nominee not in nominee_stats:
            won = has_prize_col and pd.notna(row.nominee_prize_year)
            nominee_stats[nominee] = {"won": bool(won), "total_noms": 0}
        nominee_stats[nominee]["total_noms"] += 1

    campaign_rows = []
    for name in campaign_names:
        if name in nominee_stats:
            info = nominee_stats[name]
            person_campaigns = campaigns_df[campaigns_df["nominee"] == name]
            best = person_campaigns.sort_values("n_nominations", ascending=False).iloc[0]
            window_years = best["year_end"] - best["year_start"] + 1
            window_rate = best["n_nominations"] / window_years
            nom_years = nominee_year_counts[nominee_year_counts["nominee_name"] == name]
            burst_yr_set = set(range(int(best["year_start"]), int(best["year_end"]) + 1))
            baseline = nom_years[~nom_years["year"].isin(burst_yr_set)]["count"]
            baseline_rate = float(baseline.mean()) if len(baseline) > 0 else 0
            campaign_rows.append({
                "name": name, "total_noms": info["total_noms"], "won": info["won"],
                "campaign_noms": int(best["n_nominations"]),
                "campaign_years": f"{int(best['year_start'])}-{int(best['year_end'])}",
                "campaign_nominators": int(best["n_unique_nominators"]),
                "baseline_rate": round(baseline_rate, 1),
                "burst_rate": round(window_rate, 1),
            })

    campaign_table = pd.DataFrame(campaign_rows)
    if campaign_table.empty:
        return {"error": "Could not match campaign nominees to dataset."}

    # --- Step 3: Matched control group with temporal controls ---
    # Compute first nomination year per nominee for era matching
    nominee_first_year = (
        combined_df.groupby("nominee_name")["year"].min().to_dict()
    )

    non_campaign = {name: info for name, info in nominee_stats.items()
                    if name not in campaign_names}
    non_campaign_list = list(non_campaign.items())
    control_names = set()
    for _, crow in campaign_table.iterrows():
        target_noms = crow["total_noms"]
        target_name = crow["name"]
        target_year = nominee_first_year.get(target_name, 1930)
        margin = max(2, int(target_noms * 0.3))
        lo, hi = target_noms - margin, target_noms + margin

        # Match on nomination count AND era (first nomination year +/- 15)
        matches = []
        for name, info in non_campaign_list:
            if name in control_names:
                continue
            if not (lo <= info["total_noms"] <= hi):
                continue
            ctrl_year = nominee_first_year.get(name, 1930)
            year_diff = abs(ctrl_year - target_year)
            if year_diff > 15:
                continue
            # Score: weighted combination of nomination distance + era distance
            nom_dist = abs(info["total_noms"] - target_noms)
            score = nom_dist + 0.1 * year_diff
            matches.append((name, score))
        matches.sort(key=lambda x: x[1])
        for m, _ in matches[:3]:
            control_names.add(m)

    control_rows = []
    for name in control_names:
        info = nominee_stats[name]
        control_rows.append({"name": name, "total_noms": info["total_noms"], "won": info["won"]})
    control_table = pd.DataFrame(control_rows) if control_rows else pd.DataFrame(
        columns=["name", "total_noms", "won"])

    # --- Step 4: Bayesian analyses ---
    draws = sample_kwargs.get("draws", 1000)
    tune = sample_kwargs.get("tune", 500)
    chains = sample_kwargs.get("chains", 2)

    n_campaign = len(campaign_table)
    n_campaign_won = int(campaign_table["won"].sum())
    n_control = len(control_table)
    n_control_won = int(control_table["won"].sum()) if n_control > 0 else 0

    # Beta-Binomial (replaces Fisher's exact)
    bb_idata, bb_summary = None, {}
    if n_control > 0:
        bb_idata, bb_summary = bayesian_beta_binomial(
            n_campaign, n_campaign_won, n_control, n_control_won,
            ("campaign", "control"), draws=draws, tune=tune, chains=chains)

    # Hierarchical logistic (replaces CMH)
    first_lo = min(min_nominations, 5)
    bins = [(first_lo, 10), (11, 20), (21, 50), (51, 500)]
    bin_labels = [f"{first_lo}-10", "11-20", "21-50", "51+"]
    if first_lo > 10:
        bins = [(first_lo, 20), (21, 50), (51, 500)]
        bin_labels = [f"{first_lo}-20", "21-50", "51+"]

    strata_data = []
    bin_breakdown = []
    for stratum_idx, ((lo, hi), label) in enumerate(zip(bins, bin_labels)):
        c_in_bin = campaign_table[(campaign_table["total_noms"] >= lo) &
                                  (campaign_table["total_noms"] <= hi)]
        ctrl_in_bin = control_table[(control_table["total_noms"] >= lo) &
                                    (control_table["total_noms"] <= hi)] if n_control > 0 else pd.DataFrame()
        n_c = len(c_in_bin)
        n_ctrl = len(ctrl_in_bin)
        c_won = int(c_in_bin["won"].sum()) if n_c > 0 else 0
        ctrl_won = int(ctrl_in_bin["won"].sum()) if n_ctrl > 0 else 0

        bin_breakdown.append({
            "nom_range": label, "campaign_n": n_c, "campaign_won": c_won,
            "campaign_rate": float(c_in_bin["won"].mean()) if n_c > 0 else 0,
            "control_n": n_ctrl, "control_won": ctrl_won,
            "control_rate": float(ctrl_in_bin["won"].mean()) if n_ctrl > 0 else 0,
        })

        for _, r in c_in_bin.iterrows():
            strata_data.append({"group": 1, "outcome": int(r["won"]), "stratum": stratum_idx})
        for _, r in ctrl_in_bin.iterrows():
            strata_data.append({"group": 0, "outcome": int(r["won"]), "stratum": stratum_idx})

    hl_idata, hl_summary = None, {}
    if len(strata_data) >= 6:
        hl_idata, hl_summary = bayesian_hierarchical_logistic(
            strata_data, draws=draws, tune=tune, chains=chains)

    campaign_mean_noms = float(campaign_table["total_noms"].mean())
    control_mean_noms = float(control_table["total_noms"].mean()) if n_control > 0 else 0

    n_raw = len(raw_campaigns_df["nominee"].unique())
    n_filtered = len(campaign_names)

    # Conjugate BF10 (no MCMC)
    conjugate = None
    if n_control > 0:
        conjugate = conjugate_beta_binomial_bf(
            n_campaign, n_campaign_won, n_control, n_control_won,
            ("campaign", "control"))

    return {
        "campaign_table": campaign_table.sort_values("total_noms", ascending=False),
        "control_table": control_table.sort_values("total_noms", ascending=False) if n_control > 0 else control_table,
        "n_campaign": n_campaign, "n_campaign_won": n_campaign_won,
        "campaign_win_rate": n_campaign_won / n_campaign if n_campaign > 0 else 0,
        "n_control": n_control, "n_control_won": n_control_won,
        "control_win_rate": n_control_won / n_control if n_control > 0 else 0,
        "bb_idata": bb_idata, "bb_summary": bb_summary,
        "hl_idata": hl_idata, "hl_summary": hl_summary,
        "conjugate": conjugate,
        "campaign_mean_noms": campaign_mean_noms,
        "control_mean_noms": control_mean_noms,
        "bin_breakdown": pd.DataFrame(bin_breakdown),
        "n_raw_campaigns": n_raw, "n_relative_campaigns": n_filtered,
    }


# ---------------------------------------------------------------------------
# ANALYSIS 8: BURST DECOMPOSITION — BAYESIAN
# ---------------------------------------------------------------------------

def compute_burst_decomposition(combined_df, precomputed, min_nominations=5,
                                 year_window=3, **sample_kwargs):
    from bayesian_models import (bayesian_beta_binomial,
                                  bayesian_hierarchical_logistic,
                                  bayesian_logistic_single_predictor,
                                  bayesian_logistic_regression,
                                  conjugate_beta_binomial_bf)

    raw_campaigns_df = detect_campaigns(combined_df, min_nominations=min_nominations,
                                        year_window=year_window)
    if raw_campaigns_df.empty:
        return {"error": "No bursts detected with current thresholds."}

    nominee_year_counts = combined_df.groupby(
        ["nominee_name", "year"]).size().reset_index(name="count")

    filtered_campaigns = []
    for _, row in raw_campaigns_df.iterrows():
        nominee = row["nominee"]
        nom_years = nominee_year_counts[nominee_year_counts["nominee_name"] == nominee]
        if nom_years.empty:
            continue
        window_years = row["year_end"] - row["year_start"] + 1
        window_annual_rate = row["n_nominations"] / window_years
        burst_year_set = set(range(int(row["year_start"]), int(row["year_end"]) + 1))
        baseline = nom_years[~nom_years["year"].isin(burst_year_set)]["count"]
        if len(baseline) <= 1:
            filtered_campaigns.append(row)
        else:
            base_mean = float(baseline.mean())
            base_std = float(baseline.std())
            threshold = max(base_mean + 2 * base_std, base_mean * 1.5)
            if window_annual_rate > threshold:
                filtered_campaigns.append(row)

    if not filtered_campaigns:
        return {"error": "No anomalous bursts detected."}

    campaigns_df = pd.DataFrame(filtered_campaigns)

    nominator_to_nominees = defaultdict(set)
    for row in combined_df.itertuples(index=False):
        nominator_to_nominees[row.nominator_name].add(row.nominee_name)

    has_prize_col = "nominee_prize_year" in combined_df.columns
    nominee_stats = {}
    for row in combined_df.itertuples(index=False):
        nominee = row.nominee_name
        if nominee not in nominee_stats:
            won = has_prize_col and pd.notna(row.nominee_prize_year)
            nominee_stats[nominee] = {"won": bool(won), "total_noms": 0}
        nominee_stats[nominee]["total_noms"] += 1

    burst_rows = []
    best_bursts = campaigns_df.sort_values("n_nominations", ascending=False).drop_duplicates(
        subset="nominee", keep="first")

    for _, burst in best_bursts.iterrows():
        nominee = burst["nominee"]
        if nominee not in nominee_stats:
            continue
        burst_year_set = set(range(int(burst["year_start"]), int(burst["year_end"]) + 1))
        burst_noms_df = combined_df[
            (combined_df["nominee_name"] == nominee) &
            (combined_df["year"].isin(burst_year_set))
        ]
        burst_nominators = list(burst_noms_df["nominator_name"].unique())
        n_burst_nominators = len(burst_nominators)

        if n_burst_nominators < 2:
            burst_rows.append({
                "nominee": nominee, "won": nominee_stats[nominee]["won"],
                "total_noms": nominee_stats[nominee]["total_noms"],
                "burst_noms": int(burst["n_nominations"]),
                "n_burst_nominators": n_burst_nominators, "diversity": 1,
                "density": 0.0, "connected_frac": 0.0, "n_countries": 1,
                "burst_type": "independent",
            })
            continue

        co_activity = nx.Graph()
        co_activity.add_nodes_from(burst_nominators)
        for i in range(len(burst_nominators)):
            for j in range(i + 1, len(burst_nominators)):
                n1, n2 = burst_nominators[i], burst_nominators[j]
                shared = (nominator_to_nominees.get(n1, set()) &
                          nominator_to_nominees.get(n2, set())) - {nominee}
                if shared:
                    co_activity.add_edge(n1, n2, weight=len(shared))

        connected_count = sum(1 for n in burst_nominators if co_activity.degree(n) > 0)
        connected_frac = connected_count / n_burst_nominators
        n_nodes = co_activity.number_of_nodes()
        n_edges = co_activity.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
        density = n_edges / max_edges if max_edges > 0 else 0

        if n_edges > 0 and n_nodes > 2:
            try:
                from networkx.algorithms.community import louvain_communities
                comms = louvain_communities(co_activity, seed=42)
                diversity = len(comms)
            except Exception:
                diversity = 1
        elif n_edges == 0:
            diversity = n_nodes
        else:
            diversity = max(1, n_nodes)

        if "nominator_country" in burst_noms_df.columns:
            countries = burst_noms_df["nominator_country"].dropna().unique()
            n_countries = len([c for c in countries if c != "Unknown"])
        else:
            n_countries = 0

        if connected_frac == 0:
            burst_type = "independent"
        elif connected_frac < 0.5:
            burst_type = "mixed"
        else:
            burst_type = "coordinated"

        burst_rows.append({
            "nominee": nominee, "won": nominee_stats[nominee]["won"],
            "total_noms": nominee_stats[nominee]["total_noms"],
            "burst_noms": int(burst["n_nominations"]),
            "n_burst_nominators": n_burst_nominators, "diversity": diversity,
            "density": round(density, 4), "connected_frac": round(connected_frac, 3),
            "n_countries": n_countries, "burst_type": burst_type,
        })

    if not burst_rows:
        return {"error": "Could not compute burst decomposition."}

    burst_table = pd.DataFrame(burst_rows)

    independent = burst_table[burst_table["connected_frac"] == 0]
    connected = burst_table[burst_table["connected_frac"] > 0]
    n_independent = len(independent)
    n_connected = len(connected)
    independent_win_rate = float(independent["won"].mean()) if n_independent > 0 else 0
    connected_win_rate = float(connected["won"].mean()) if n_connected > 0 else 0

    draws = sample_kwargs.get("draws", 1000)
    tune = sample_kwargs.get("tune", 500)
    chains = sample_kwargs.get("chains", 2)

    # Beta-Binomial (replaces Fisher)
    bb_idata, bb_summary = None, {}
    if n_independent > 0 and n_connected > 0:
        i_won = int(independent["won"].sum())
        c_won = int(connected["won"].sum())
        bb_idata, bb_summary = bayesian_beta_binomial(
            n_connected, c_won, n_independent, i_won,
            ("connected", "independent"), draws=draws, tune=tune, chains=chains)

    # Hierarchical logistic (replaces CMH)
    first_lo = min(min_nominations, 5)
    bins = [(first_lo, 10), (11, 20), (21, 50), (51, 500)]
    bin_labels = [f"{first_lo}-10", "11-20", "21-50", "51+"]
    if first_lo > 10:
        bins = [(first_lo, 20), (21, 50), (51, 500)]
        bin_labels = [f"{first_lo}-20", "21-50", "51+"]

    strata_data = []
    bin_breakdown = []
    for stratum_idx, ((lo, hi), label) in enumerate(zip(bins, bin_labels)):
        ind_in_bin = independent[(independent["total_noms"] >= lo) & (independent["total_noms"] <= hi)]
        con_in_bin = connected[(connected["total_noms"] >= lo) & (connected["total_noms"] <= hi)]
        n_ind = len(ind_in_bin)
        n_con = len(con_in_bin)
        ind_won = int(ind_in_bin["won"].sum()) if n_ind > 0 else 0
        con_won = int(con_in_bin["won"].sum()) if n_con > 0 else 0
        bin_breakdown.append({
            "nom_range": label, "outsider_n": n_ind, "outsider_won": ind_won,
            "outsider_rate": float(ind_in_bin["won"].mean()) if n_ind > 0 else 0,
            "insider_n": n_con, "insider_won": con_won,
            "insider_rate": float(con_in_bin["won"].mean()) if n_con > 0 else 0,
        })
        for _, r in con_in_bin.iterrows():
            strata_data.append({"group": 1, "outcome": int(r["won"]), "stratum": stratum_idx})
        for _, r in ind_in_bin.iterrows():
            strata_data.append({"group": 0, "outcome": int(r["won"]), "stratum": stratum_idx})

    hl_idata, hl_summary = None, {}
    if len(strata_data) >= 6:
        hl_idata, hl_summary = bayesian_hierarchical_logistic(
            strata_data, draws=draws, tune=tune, chains=chains)

    # Bayesian logistic: n_countries vs winning (replaces point-biserial)
    bl_idata, bl_summary = None, {}
    if len(burst_table) >= 6:
        try:
            bl_idata, bl_summary = bayesian_logistic_single_predictor(
                burst_table["n_countries"].values,
                burst_table["won"].astype(int).values,
                "n_countries", draws=draws, tune=tune, chains=chains)
        except Exception:
            pass

    # --- Conjugate BF10 (no MCMC) ---
    conjugate = None
    if n_independent > 0 and n_connected > 0:
        i_won = int(independent["won"].sum())
        c_won = int(connected["won"].sum())
        conjugate = conjugate_beta_binomial_bf(
            n_connected, c_won, n_independent, i_won,
            ("connected", "independent"))

    # --- Dose-response logistic: connected_frac + log(total_noms) ---
    dr_idata, dr_summary = None, {}
    if len(burst_table) >= 10:
        try:
            X_dr = np.column_stack([
                burst_table["connected_frac"].values,
                np.log1p(burst_table["total_noms"].values)])
            dr_idata, dr_summary = bayesian_logistic_regression(
                X_dr, burst_table["won"].astype(int).values,
                ["connected_frac", "log_total_noms"],
                "dose_response", draws=draws, tune=tune, chains=chains)
        except Exception:
            pass

    # --- Category column: join category from combined_df ---
    nominee_category = (
        combined_df.groupby("nominee_name")["category"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    burst_table["category"] = burst_table["nominee"].map(nominee_category)

    # --- Science-only subset: Physics, Chemistry, Physiology or Medicine ---
    # Only meaningful when multiple categories are present (cross-category data)
    science_cats = {"Physics", "Chemistry", "Physiology or Medicine"}
    science_only = None
    unique_cats = set(burst_table["category"].dropna().unique())
    sci_table = burst_table[burst_table["category"].isin(science_cats)]
    sci_ind = sci_table[sci_table["connected_frac"] == 0]
    sci_con = sci_table[sci_table["connected_frac"] > 0]
    if len(unique_cats) > 1 and len(sci_ind) >= 2 and len(sci_con) >= 2:
        sci_i_won = int(sci_ind["won"].sum())
        sci_c_won = int(sci_con["won"].sum())
        science_only = conjugate_beta_binomial_bf(
            len(sci_con), sci_c_won, len(sci_ind), sci_i_won,
            ("connected", "independent"))

    # --- Per-category BF10 ---
    per_category = {}
    for cat, cat_df in burst_table.groupby("category"):
        if pd.isna(cat):
            continue
        cat_ind = cat_df[cat_df["connected_frac"] == 0]
        cat_con = cat_df[cat_df["connected_frac"] > 0]
        if len(cat_ind) >= 4 and len(cat_con) >= 4:
            cat_i_won = int(cat_ind["won"].sum())
            cat_c_won = int(cat_con["won"].sum())
            per_category[cat] = conjugate_beta_binomial_bf(
                len(cat_con), cat_c_won, len(cat_ind), cat_i_won,
                ("connected", "independent"))

    return {
        "burst_table": burst_table.sort_values("burst_noms", ascending=False),
        "n_independent": n_independent, "n_connected": n_connected,
        "independent_win_rate": independent_win_rate,
        "connected_win_rate": connected_win_rate,
        "bb_idata": bb_idata, "bb_summary": bb_summary,
        "hl_idata": hl_idata, "hl_summary": hl_summary,
        "bl_idata": bl_idata, "bl_summary": bl_summary,
        "bin_breakdown": pd.DataFrame(bin_breakdown),
        "conjugate": conjugate,
        "dr_idata": dr_idata, "dr_summary": dr_summary,
        "science_only": science_only,
        "per_category": per_category,
    }


# ---------------------------------------------------------------------------
# ANALYSIS 9: CENTRALITY PREDICTS WINNERS — BAYESIAN
# ---------------------------------------------------------------------------

def compute_centrality_prediction(combined_df, precomputed, **sample_kwargs):
    from bayesian_models import bayesian_logistic_regression, posterior_predictive_roc
    import arviz as az

    laureate_lookup = build_laureate_lookup(precomputed)
    df = combined_df.copy()
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    snapshot_years = list(range(1910, 1970, 5))
    all_observations = []

    for snap_year in snapshot_years:
        snap_df = df[df["year"] <= snap_year]
        if len(snap_df) < 50:
            continue
        snap_nom_stats = defaultdict(lambda: defaultdict(int))
        for row in snap_df.itertuples(index=False):
            snap_nom_stats[row.nominee_name][row.nominator_name] += 1
        G = nx.DiGraph()
        for row in snap_df.itertuples(index=False):
            n1, n2 = row.nominator_name, row.nominee_name
            if G.has_edge(n1, n2):
                G[n1][n2]["weight"] += 1
            else:
                G.add_edge(n1, n2, weight=1)
        if G.number_of_nodes() < 10:
            continue
        nominees_in_graph = [n for n in G.nodes if G.in_degree(n) > 0]
        if len(nominees_in_graph) < 10:
            continue
        in_degrees = dict(G.in_degree(weight="weight"))
        try:
            pagerank = nx.pagerank(G, weight="weight", max_iter=100)
        except nx.PowerIterationFailedConvergence:
            pagerank = {n: 1.0 / max(1, G.number_of_nodes()) for n in G.nodes}

        snap_nominee_to_nors = defaultdict(set)
        for row in snap_df.itertuples(index=False):
            snap_nominee_to_nors[row.nominee_name].add(row.nominator_name)
        nom_shared = defaultdict(lambda: defaultdict(int))
        for nom, nors in snap_nominee_to_nors.items():
            nors_l = sorted(nors)
            for i in range(len(nors_l)):
                for j in range(i + 1, len(nors_l)):
                    nom_shared[nors_l[i]][nors_l[j]] += 1
                    nom_shared[nors_l[j]][nors_l[i]] += 1

        for nominee in nominees_in_graph:
            won = 0
            laureate = match_nominator_to_laureate(nominee, laureate_lookup)
            if laureate and snap_year < laureate["year_won"] <= snap_year + 10:
                won = 1
            nc = snap_nom_stats[nominee]
            total_noms = sum(nc.values())
            breadth = len(nc)
            nom_degs = [G.degree(n, weight="weight") for n in nc if n in G]
            reach = float(np.mean(nom_degs)) if nom_degs else 0.0
            if total_noms > 0:
                shares = [c / total_noms for c in nc.values()]
                concentration = sum(s ** 2 for s in shares)
            else:
                concentration = 1.0
            noms_list = list(nc.keys())
            if len(noms_list) > 2:
                sub_g = nx.Graph()
                for n in noms_list:
                    sub_g.add_node(n)
                for i, n1 in enumerate(noms_list):
                    for n2 in noms_list[i + 1:]:
                        w = nom_shared.get(n1, {}).get(n2, 0)
                        if w > 0:
                            sub_g.add_edge(n1, n2, weight=w)
                if sub_g.number_of_edges() > 0:
                    try:
                        from networkx.algorithms.community import louvain_communities
                        communities = louvain_communities(sub_g, seed=42)
                        diversity = len(communities)
                    except Exception:
                        diversity = len(noms_list)
                else:
                    diversity = len(noms_list)
            else:
                diversity = max(1, len(noms_list))
            # Mean PageRank of nominators
            nominator_prs = [pagerank.get(n, 0) for n in nc.keys()]
            mean_nominator_pagerank = float(np.mean(nominator_prs)) if nominator_prs else 0.0

            all_observations.append({
                "snapshot": snap_year, "nominee": nominee,
                "in_degree": in_degrees.get(nominee, 0),
                "pagerank": pagerank.get(nominee, 0),
                "nominator_pagerank": round(mean_nominator_pagerank, 8),
                "breadth": breadth, "diversity": diversity,
                "reach": round(reach, 2), "concentration": round(concentration, 4),
                "won": won,
            })

    if not all_observations:
        return {"error": "Not enough data for centrality prediction"}

    feature_table = pd.DataFrame(all_observations)

    in_degree_features = ["in_degree"]
    structural_features = ["breadth", "diversity", "reach", "concentration"]
    full_features = ["in_degree", "pagerank", "breadth", "diversity", "reach", "concentration"]
    full_nompr_features = ["in_degree", "pagerank", "nominator_pagerank",
                           "breadth", "diversity", "reach", "concentration"]

    draws = sample_kwargs.get("draws", 1000)
    tune = sample_kwargs.get("tune", 500)
    chains = sample_kwargs.get("chains", 2)

    y_all = feature_table["won"].values

    # Fit four Bayesian logistic regression models
    model_configs = {
        "in_degree_only": in_degree_features,
        "structural": structural_features,
        "full": full_features,
        "full_nompr": full_nompr_features,
    }

    model_results = {}
    idatas = {}
    for model_name, feats in model_configs.items():
        X = feature_table[feats].values
        idata, summary = bayesian_logistic_regression(
            X, y_all, feats, model_name,
            draws=draws, tune=tune, chains=chains)
        model_results[model_name] = summary
        idatas[model_name] = idata

    # WAIC/LOO model comparison
    compare_dict = {}
    for name, idata in idatas.items():
        try:
            compare_dict[name] = idata
        except Exception:
            pass

    comparison_df = None
    try:
        comparison_df = az.compare(compare_dict, ic="waic")
    except Exception:
        try:
            comparison_df = az.compare(compare_dict, ic="loo")
        except Exception:
            pass

    # Posterior predictive ROC for each model
    roc_data = {}
    for model_name, feats in model_configs.items():
        X = feature_table[feats].values
        roc_result = posterior_predictive_roc(
            idatas[model_name], X, y_all, feats, n_draws=200)
        roc_data[model_name] = roc_result

    return {
        "model_results": model_results,
        "idatas": idatas,
        "roc_data": roc_data,
        "comparison_df": comparison_df,
        "feature_table": feature_table,
        "n_snapshots": len(set(feature_table["snapshot"])),
    }


# ---------------------------------------------------------------------------
# NETWORK STATISTICS
# ---------------------------------------------------------------------------

def network_summary(G):
    stats = {"Nodes": G.number_of_nodes(), "Edges": G.number_of_edges(),
             "Density": round(nx.density(G), 4)}
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
        components = list(nx.connected_components(G))
        stats["Components"] = len(components)
        stats["Largest component"] = len(max(components, key=len)) if components else 0
    return stats, name_stats


# ---------------------------------------------------------------------------
# STREAMLIT PAGE — RENDERING
# ---------------------------------------------------------------------------

@st.cache_resource
def _cached_build_graph(df_hash, _df, network_type, country_filter, _precomputed):
    if network_type == "Nominator -> Nominee":
        return build_nomination_graph(_df, country=country_filter)
    elif network_type == "Co-nomination (shared nominators)":
        return build_conomination_graph(_df)
    elif network_type == "Cross-category Combined":
        return build_combined_nomination_graph(_df, _precomputed)
    return nx.Graph()


@st.cache_data
def _cached_visualize(graph_key, _G, title, min_edge_weight, color_by):
    html_path = visualize_graph(_G, title=title, min_edge_weight=min_edge_weight, color_by=color_by)
    if html_path:
        with open(html_path, "r") as f:
            html_content = f.read()
        os.unlink(html_path)
        return html_content
    return None


def _df_hash(df):
    import hashlib
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    h.update(",".join(df.columns).encode())
    h.update(pd.util.hash_pandas_object(df).values.tobytes())
    return h.hexdigest()


def _render_graph_visualization(G, network_type, min_weight, is_cross_category):
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
    stats, name_stats = network_summary(G)
    cols = st.columns(len(stats))
    for i, (k, v) in enumerate(stats.items()):
        cols[i].metric(k, v)
    if name_stats:
        for label, name in name_stats.items():
            st.markdown(f"**{label}:** {name}")
    if network_type == "Nominator -> Nominee":
        st.markdown("**Directed graph**: arrows point from nominator to nominee. "
                     "Node size = total nominations received. "
                     "Edge thickness = number of times that nominator proposed that nominee.")
    elif network_type == "Co-nomination (shared nominators)":
        st.markdown("**Undirected graph**: nominees are linked when they share nominators. "
                     "Clusters reveal communities of nominees championed by the same people.")
    elif is_cross_category:
        st.markdown("**Undirected graph**: all 5 Nobel categories combined. "
                     "Node color = discipline. Gold border = laureate.")
    if is_cross_category:
        categories_in_graph = set()
        for node in G.nodes:
            c = G.nodes[node].get("category", "")
            if c:
                categories_in_graph.add(c)
        legend_items = []
        for cat in sorted(categories_in_graph):
            color = DISCIPLINE_COLORS.get(cat, "#999999")
            legend_items.append(f'<span style="color:{color}; font-size:20px;">&#9679;</span> {cat}')
        legend_items.append('<span style="color:#FFD700; font-size:20px;">&#9679;</span> Laureate (gold border)')
        st.markdown("**Node color = discipline:** " + " &nbsp;&nbsp; ".join(legend_items),
                     unsafe_allow_html=True)
    else:
        countries_in_graph = set()
        for node in G.nodes:
            c = G.nodes[node].get("country", "Unknown")
            if c:
                countries_in_graph.add(c)
        legend_items = []
        for country in sorted(countries_in_graph):
            color = COUNTRY_COLORS.get(country, "#cccccc")
            legend_items.append(f'<span style="color:{color}; font-size:20px;">&#9679;</span> {country}')
        if legend_items:
            st.markdown("**Node color = country:** " + " &nbsp;&nbsp; ".join(legend_items),
                         unsafe_allow_html=True)
    if network_type == "Co-nomination (shared nominators)":
        if st.checkbox("Run community detection (Louvain)"):
            try:
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(G, weight="weight", seed=42)
                for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:10]):
                    st.write(f"**Community {i+1}** ({len(comm)} members): "
                             f"{', '.join(sorted(comm)[:10])}{'...' if len(comm) > 10 else ''}")
            except ImportError:
                st.warning("Louvain requires networkx >= 2.8")


def _render_campaigns(df, min_noms, campaign_window):
    st.subheader("Campaign Detection")
    campaigns = detect_campaigns(df, min_nominations=min_noms, year_window=campaign_window)
    if len(campaigns) > 0:
        display_cols = ["nominee", "year_start", "year_end", "n_nominations", "n_unique_nominators"]
        st.dataframe(campaigns[display_cols], hide_index=True)
        _csv_download_button(campaigns[display_cols], "campaigns.csv", key="campaigns_csv")
    else:
        st.info("No campaigns detected with current thresholds.")


def _render_paper_analyses(df, combined_df, precomputed, has_combined, run_lcc):
    import matplotlib.pyplot as plt
    st.subheader("Paper Analyses (Gallotti & De Domenico, 2019)")
    st.caption("Inspired by *Effects of homophily and academic reputation "
               "in the nomination and selection of Nobel laureates*.")
    st.markdown("#### Laureate Endorsement Effect")
    st.caption("Do nominees endorsed by past laureates win at higher rates?")
    analysis_df = combined_df if has_combined else df
    effect = compute_endorsement_effect(analysis_df, precomputed)
    col1, col2, col3 = st.columns(3)
    col1.metric("Endorsed win rate", f"{effect['endorsed_rate']:.1%}",
                help=f"{effect['endorsed_won']}/{effect['endorsed_total']}")
    col2.metric("Non-endorsed win rate", f"{effect['not_endorsed_rate']:.1%}",
                help=f"{effect['not_endorsed_won']}/{effect['not_endorsed_total']}")
    ratio_str = f"{effect['ratio']:.1f}x" if effect['ratio'] != float("inf") else "N/A"
    col3.metric("Ratio", ratio_str)
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(["Endorsed\nby laureate", "Not endorsed"],
                  [effect["endorsed_rate"] * 100, effect["not_endorsed_rate"] * 100],
                  color=["#FFD700", "#999999"], edgecolor="black")
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
                st.caption(f"LCC: {lcc_result['lcc_size']}/{lcc_result['graph_size']} nodes. "
                           f"Laureates: {lcc_result['n_laureates']}.")


# ---------------------------------------------------------------------------
# RENDER NETWORK PAGE (entry point)
# ---------------------------------------------------------------------------

def render_network_page(df, precomputed=None, combined_df=None,
                        category=None, sample_kwargs=None):
    if not HAS_STREAMLIT:
        print("Streamlit not available.")
        return

    if sample_kwargs is None:
        sample_kwargs = {"draws": 1000, "tune": 500, "chains": 2}

    has_combined = combined_df is not None and not combined_df.empty
    has_precomputed = precomputed is not None and len(precomputed) > 0

    if has_combined and category:
        cat_combined = combined_df[combined_df["category"] == category].copy()
    else:
        cat_combined = combined_df

    # --- Sidebar: Network Controls ---
    st.sidebar.divider()
    st.sidebar.header("Network Controls")
    network_options = ["Nominator -> Nominee", "Co-nomination (shared nominators)"]
    if has_combined:
        network_options.append("Cross-category Combined")
    network_type = st.sidebar.selectbox("Network type", network_options)
    is_cross_category = network_type == "Cross-category Combined"
    default_weight = 2 if is_cross_category else 1
    min_weight = st.sidebar.slider("Min edge weight", 1, 10, default_weight)
    if is_cross_category and min_weight == 1:
        st.sidebar.warning("Min weight 1 produces ~15K nodes -- rendering may be slow.")
    country_filter = None
    if network_type == "Nominator -> Nominee" and "nominee_country" in df.columns:
        country_options = ["All"] + sorted(df["nominee_country"].dropna().unique().tolist())
        country_filter = st.sidebar.selectbox("Filter nominee country", country_options)
        if country_filter == "All":
            country_filter = None

    # --- Sidebar: Analyses (grouped into Bayesian vs Descriptive) ---
    st.sidebar.divider()
    st.sidebar.header("Analyses")

    # Bayesian analyses (PyMC + ArviZ)
    bayesian_options = ["None"]
    if has_combined and has_precomputed:
        bayesian_options.extend([
            "Near-Miss Analysis",
            "Campaign Success Rate",
            "Burst Decomposition",
            "Centrality Predicts Winners",
        ])

    if len(bayesian_options) > 1:
        st.sidebar.subheader("Bayesian (PyMC)")
        bayesian_selection = st.sidebar.radio(
            "Bayesian analysis", bayesian_options,
            key="bayesian_select", label_visibility="collapsed")
    else:
        bayesian_selection = "None"

    # Descriptive / network analyses (no MCMC)
    descriptive_options = ["None"]
    if has_combined:
        descriptive_options.append("Temporal Evolution")
    if has_precomputed:
        descriptive_options.append("Endorsement & LCC")
    if has_combined and has_precomputed:
        descriptive_options.append("Three Degrees of Influence")
    descriptive_options.append("Campaign Detection")
    descriptive_options.append("Raw Edge Data")

    st.sidebar.subheader("Descriptive")
    descriptive_selection = st.sidebar.radio(
        "Descriptive analysis", descriptive_options,
        key="descriptive_select", label_visibility="collapsed")

    # Only one group can be active at a time
    if bayesian_selection != "None" and descriptive_selection != "None":
        # Bayesian takes priority if both changed; use session state to detect
        _prev_b = st.session_state.get("_prev_bayesian", "None")
        _prev_d = st.session_state.get("_prev_descriptive", "None")
        if bayesian_selection != _prev_b:
            analysis_selection = bayesian_selection
        elif descriptive_selection != _prev_d:
            analysis_selection = descriptive_selection
        else:
            analysis_selection = bayesian_selection
    elif bayesian_selection != "None":
        analysis_selection = bayesian_selection
    elif descriptive_selection != "None":
        analysis_selection = descriptive_selection
    else:
        analysis_selection = "None"
    st.session_state["_prev_bayesian"] = bayesian_selection
    st.session_state["_prev_descriptive"] = descriptive_selection

    # Per-analysis flags
    show_temporal = analysis_selection == "Temporal Evolution"
    show_paper = analysis_selection == "Endorsement & LCC"
    show_centrality = analysis_selection == "Centrality Predicts Winners"
    show_proximity = analysis_selection == "Three Degrees of Influence"
    show_near_miss = analysis_selection == "Near-Miss Analysis"
    show_campaigns = analysis_selection == "Campaign Detection"
    show_campaign_success = analysis_selection == "Campaign Success Rate"
    show_burst_decomp = analysis_selection == "Burst Decomposition"
    show_raw = analysis_selection == "Raw Edge Data"

    run_campaigns = run_lcc = False
    run_centrality = run_near_miss = run_temporal = run_proximity = False
    run_campaign_success = run_burst_decomp = False
    min_noms = 5; campaign_window = 3; near_miss_min = 10
    cs_min_noms = 5; cs_window = 3; bd_min_noms = 5; bd_window = 3

    if show_temporal:
        run_temporal = st.sidebar.button("Run Evolution Analysis", key="temporal_btn")
    elif show_paper:
        run_lcc = st.sidebar.button("Run LCC Analysis", key="lcc_btn")
    elif show_centrality:
        run_centrality = st.sidebar.button("Run Centrality Analysis", key="centrality_btn")
    elif show_proximity:
        run_proximity = st.sidebar.button("Run Proximity Analysis", key="proximity_btn")
    elif show_near_miss:
        near_miss_min = st.sidebar.slider("Min nominations", 5, 30, 10, key="near_miss_min")
        run_near_miss = st.sidebar.button("Run Near-Miss Analysis", key="near_miss_btn")
    elif show_campaigns:
        min_noms = st.sidebar.slider("Min nominations", 3, 15, 5, key="campaign_min")
        campaign_window = st.sidebar.slider("Year window", 1, 5, 3, key="campaign_window")
        run_campaigns = st.sidebar.button("Detect campaigns")
    elif show_campaign_success:
        cs_min_noms = st.sidebar.slider("Min nominations (burst)", 3, 15, 5, key="cs_min_noms")
        cs_window = st.sidebar.slider("Year window", 1, 5, 3, key="cs_window")
        run_campaign_success = st.sidebar.button("Run Campaign Success Analysis", key="cs_btn")
    elif show_burst_decomp:
        bd_min_noms = st.sidebar.slider("Min nominations", 3, 15, 5, key="bd_min_noms")
        bd_window = st.sidebar.slider("Year window", 1, 5, 3, key="bd_window")
        run_burst_decomp = st.sidebar.button("Run Burst Decomposition", key="bd_btn")

    # --- Main area ---
    st.header("Nomination Networks")
    if network_type == "Nominator -> Nominee":
        G = _cached_build_graph(_df_hash(df), df, network_type, country_filter, None)
    elif network_type == "Co-nomination (shared nominators)":
        G = _cached_build_graph(_df_hash(df), df, network_type, None, None)
    elif is_cross_category:
        with st.spinner("Building cross-category combined network..."):
            G = _cached_build_graph(_df_hash(combined_df), combined_df, network_type, None, precomputed)

    _render_graph_visualization(G, network_type, min_weight, is_cross_category)

    if analysis_selection != "None":
        st.divider()

    if show_campaigns and run_campaigns:
        _render_campaigns(df, min_noms, campaign_window)
    if show_paper:
        _render_paper_analyses(df, cat_combined, precomputed, has_combined, run_lcc)
    if show_raw:
        st.subheader("Raw Edge Data")
        st.dataframe(df, hide_index=True)
        _csv_download_button(df, "nomination_edges.csv", key="raw_edges_csv")

    adv_flags = {
        "show_centrality": show_centrality, "run_centrality": run_centrality,
        "show_near_miss": show_near_miss, "run_near_miss": run_near_miss,
        "near_miss_min": near_miss_min,
        "show_temporal": show_temporal, "run_temporal": run_temporal,
        "show_proximity": show_proximity, "run_proximity": run_proximity,
        "show_campaign_success": show_campaign_success,
        "run_campaign_success": run_campaign_success,
        "cs_min_noms": cs_min_noms, "cs_window": cs_window,
        "show_burst_decomp": show_burst_decomp,
        "run_burst_decomp": run_burst_decomp,
        "bd_min_noms": bd_min_noms, "bd_window": bd_window,
        "sample_kwargs": sample_kwargs,
    }
    _render_advanced_analyses(combined_df, cat_combined, precomputed, adv_flags)


# ---------------------------------------------------------------------------
# RENDERING HELPER: CONJUGATE BAYES FACTOR
# ---------------------------------------------------------------------------

def _render_bayes_factor(conjugate, title=None, key_prefix="bf"):
    """
    Render conjugate Beta-Binomial results + Bayes Factor in an expander.

    Parameters
    ----------
    conjugate : dict
        Output of conjugate_beta_binomial_bf().
    title : str or None
        Expander title. Default: "Bayes Factor Analysis".
    key_prefix : str
        Unique prefix for Streamlit widget keys.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import beta as beta_dist

    if conjugate is None:
        return

    if title is None:
        g1, g2 = conjugate["group_names"]
        title = f"Bayes Factor: {g1} vs {g2}"

    with st.expander(title, expanded=True):
        bf10 = conjugate["bf10"]
        interp = conjugate["bf10_interpretation"]
        g1, g2 = conjugate["group_names"]

        # BF10 metric row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BF10", f"{bf10:.2f}" if bf10 < 1000 else f"{bf10:.0f}",
                     help=f"Bayes Factor (separate rates vs common rate): {interp}")
        col2.metric(f"P({g1})", f"{conjugate['p1_mean']:.3f}",
                     help=f"94% HDI: [{conjugate['p1_hdi'][0]:.3f}, {conjugate['p1_hdi'][1]:.3f}]")
        col3.metric(f"P({g2})", f"{conjugate['p2_mean']:.3f}",
                     help=f"94% HDI: [{conjugate['p2_hdi'][0]:.3f}, {conjugate['p2_hdi'][1]:.3f}]")
        col4.metric("Diff", f"{conjugate['diff_mean']:.3f}",
                     help=f"94% HDI: [{conjugate['diff_hdi'][0]:.3f}, {conjugate['diff_hdi'][1]:.3f}]")

        st.caption(
            f"**BF10 = {bf10:.2f}** ({interp}). "
            f"P({g1} > {g2}) = {conjugate['p_greater']:.3f}. "
            f"Based on {conjugate['n1']} {g1} ({conjugate['k1']} won) vs "
            f"{conjugate['n2']} {g2} ({conjugate['k2']} won)."
        )

        # Two-panel plot: Beta posteriors + difference histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

        # Panel 1: overlaid Beta posteriors
        x = np.linspace(0, 1, 500)
        pdf1 = beta_dist.pdf(x, conjugate["p1_alpha"], conjugate["p1_beta"])
        pdf2 = beta_dist.pdf(x, conjugate["p2_alpha"], conjugate["p2_beta"])
        ax1.plot(x, pdf1, color="#d62728", linewidth=2, label=g1)
        ax1.fill_between(x, pdf1, alpha=0.2, color="#d62728")
        ax1.plot(x, pdf2, color="#2ca02c", linewidth=2, label=g2)
        ax1.fill_between(x, pdf2, alpha=0.2, color="#2ca02c")
        ax1.set_xlabel("Win probability")
        ax1.set_ylabel("Density")
        ax1.set_title("Beta Posteriors")
        ax1.legend(fontsize=8)

        # Panel 2: difference histogram
        diff_samples = conjugate["samples1"][:50000] - conjugate["samples2"][:50000]
        ax2.hist(diff_samples, bins=80, density=True, alpha=0.7, color="#1f77b4")
        ax2.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No difference")
        lo, hi = conjugate["diff_hdi"]
        ax2.axvspan(lo, hi, alpha=0.15, color="black", label="94% HDI")
        ax2.set_xlabel(f"P({g1}) - P({g2})")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Posterior Difference (BF10={bf10:.1f})")
        ax2.legend(fontsize=7)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# RENDERING: ADVANCED ANALYSES (Bayesian edition)
# ---------------------------------------------------------------------------

def _render_advanced_analyses(combined_df, cat_combined, precomputed, flags):
    import matplotlib.pyplot as plt
    import arviz as az
    from bayesian_models import render_bayesian_diagnostics

    sample_kwargs = flags.get("sample_kwargs", {"draws": 1000, "tune": 500, "chains": 2})

    # --- Temporal Evolution (unchanged) ---
    if flags.get("show_temporal"):
        st.subheader("Temporal Network Evolution")
        with st.expander("About this analysis"):
            st.markdown(
                "**What this measures:** Cumulative nomination network built year-by-year "
                "(1901-1970), tracking GCC fraction, mean degree, clustering coefficient, "
                "and network size.")
        if flags.get("run_temporal"):
            with st.spinner("Computing temporal evolution..."):
                result = compute_temporal_evolution(combined_df)
            if "error" in result:
                st.error(result["error"])
            else:
                overall = result["overall"]
                by_category = result["by_category"]
                categories = result["category_list"]
                cat_colors = {
                    "Physics": "#1f77b4", "Chemistry": "#2ca02c",
                    "Physiology or Medicine": "#d62728", "Medicine": "#d62728",
                    "Literature": "#9467bd", "Peace": "#ff7f0e",
                }
                fig, axes = plt.subplots(2, 2, figsize=(12, 9))
                ax = axes[0, 0]
                ax.plot(overall.index, overall["gcc_frac"], color="black", linewidth=2, label="Overall")
                for cat in categories:
                    if cat in by_category:
                        ax.plot(by_category[cat].index, by_category[cat]["gcc_frac"],
                                color=cat_colors.get(cat, "#999"), linewidth=1, alpha=0.7, label=cat)
                ax.axvline(x=1925, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylabel("GCC fraction"); ax.set_title("(a) Giant Connected Component")
                ax.legend(fontsize=7, loc="lower right")
                ax = axes[0, 1]
                ax.plot(overall.index, overall["mean_degree"], color="black", linewidth=2, label="Overall")
                for cat in categories:
                    if cat in by_category:
                        ax.plot(by_category[cat].index, by_category[cat]["mean_degree"],
                                color=cat_colors.get(cat, "#999"), linewidth=1, alpha=0.7, label=cat)
                ax.axvline(x=1925, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylabel("Mean degree"); ax.set_title("(b) Mean Degree")
                ax.legend(fontsize=7, loc="upper left")
                ax = axes[1, 0]
                ax.plot(overall.index, overall["clustering"], color="black", linewidth=2)
                ax.axvline(x=1925, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylabel("Clustering coefficient"); ax.set_xlabel("Year")
                ax.set_title("(c) Clustering Coefficient")
                ax = axes[1, 1]
                ax.plot(overall.index, overall["nodes"], color="black", linewidth=2, label="Overall")
                for cat in categories:
                    if cat in by_category:
                        ax.plot(by_category[cat].index, by_category[cat]["nodes"],
                                color=cat_colors.get(cat, "#999"), linewidth=1, alpha=0.7, label=cat)
                ax.axvline(x=1925, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylabel("Number of nodes"); ax.set_xlabel("Year")
                ax.set_title("(d) Network Size"); ax.legend(fontsize=7, loc="upper left")
                fig.tight_layout()
                st.pyplot(fig)
                _fig_download_buttons(fig, "temporal_evolution", "temporal")
                plt.close(fig)
                _csv_download_button(overall.reset_index(), "temporal_evolution.csv", key="temporal_csv")
        else:
            st.info("Click **Run Evolution Analysis** in the sidebar.")

    # --- Three Degrees of Influence (unchanged) ---
    if flags.get("show_proximity"):
        st.subheader("Three Degrees of Influence")
        with st.expander("About this analysis"):
            st.markdown(
                "**What this measures:** Shortest-path distance from each nominee to the "
                "nearest *past* laureate in the co-nomination network.")
        if flags.get("run_proximity"):
            with st.spinner("Computing proximity to laureates..."):
                result = compute_proximity_effect(cat_combined, precomputed)
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
                col3.metric("Win rate (unreachable)", f"{result['unreachable_rate'] * 100:.1f}%")
                fig, ax = plt.subplots(figsize=(7, 4))
                x_labels = by_dist["distance"].values
                win_rates = by_dist["win_rate"].values * 100
                colors = ["#2ca02c", "#98df8a", "#ffbb78", "#ff7f0e", "#d62728"]
                bars = ax.bar(x_labels, win_rates, color=colors[:len(x_labels)], edgecolor="black")
                ax.set_xlabel("Distance to nearest past laureate")
                ax.set_ylabel("Win rate (%)")
                ax.set_title("Proximity to Past Laureates and Win Rate")
                max_rate = max(win_rates) if max(win_rates) > 0 else 10
                ax.set_ylim(0, max_rate * 1.55)
                for bar, rate, row_data in zip(bars, win_rates, by_dist.itertuples()):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_rate * 0.03,
                            f"{rate:.1f}% (n={row_data.n_nominees})", ha="center", va="bottom", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                _fig_download_buttons(fig, "proximity_effect", "proximity")
                plt.close(fig)
                _csv_download_button(by_dist, "proximity_distance.csv", key="proximity_csv")
                by_decade = result.get("by_decade", {})
                if by_decade:
                    st.markdown("#### Win Rate by Distance, Stratified by Decade")
                    decade_rows = []
                    for dec in sorted(by_decade.keys()):
                        for _, r in by_decade[dec].iterrows():
                            decade_rows.append({"Decade": dec, "Distance": r["distance"],
                                                "N": r["n_nominees"], "Won": r["n_won"],
                                                "Win rate": f"{r['win_rate'] * 100:.1f}%"})
                    st.dataframe(pd.DataFrame(decade_rows), hide_index=True)

                # Dist-1 vs Dist-2 Bayes Factor comparison
                proximity_bf = result.get("proximity_bf")
                if proximity_bf is not None:
                    st.markdown("#### Bayesian Comparison: Distance 1 vs Distance 2")
                    st.caption(
                        "Compares win rates of nominees at distance 1 (directly share "
                        "a nominator with a past laureate) vs distance 2 (two hops away). "
                        "A high BF10 supports the idea that direct proximity to laureates "
                        "confers an additional advantage."
                    )
                    _render_bayes_factor(proximity_bf,
                                         title="Distance 1 vs Distance 2 Win Rates",
                                         key_prefix="prox_bf")
        else:
            st.info("Click **Run Proximity Analysis** in the sidebar.")

    # --- Near-Miss Analysis (BAYESIAN) ---
    if flags.get("show_near_miss"):
        st.subheader("Near-Miss Analysis (Bayesian)")
        with st.expander("About this analysis"):
            st.markdown(
                "Compares network metrics (breadth, diversity, reach, concentration) "
                "between Nobel winners and near-misses who were heavily nominated but "
                "never won.\n\n"
                "**Why Bayesian?** A frequentist test (Mann-Whitney U) only gives a "
                "p-value — the probability of seeing data this extreme *if* the groups "
                "were identical. The Bayesian robust t-test instead estimates the full "
                "**posterior distribution of the difference** between groups. This tells "
                "you:\n"
                "- **How large** the difference probably is (not just whether it exists)\n"
                "- **How uncertain** the estimate is (via the 94% HDI interval)\n"
                "- **The direct probability** that winners score higher, e.g. "
                "P(winners > near-misses) = 0.97\n\n"
                "The Student-t likelihood (with estimated degrees of freedom) also makes "
                "the model robust to outliers, unlike the normal assumption behind "
                "standard t-tests.")
        if flags.get("run_near_miss"):
            min_noms = flags.get("near_miss_min", 10)
            with st.spinner(f"Running Bayesian near-miss analysis (min {min_noms} nominations)..."):
                result = compute_near_miss_analysis(cat_combined, precomputed,
                                                    min_nominations=min_noms, **sample_kwargs)
            col1, col2 = st.columns(2)
            col1.metric("Near-misses", result["n_near_misses"])
            col2.metric("Comparable winners", result["n_winners"])

            metrics = ["breadth", "diversity", "reach", "concentration"]
            titles = ["Support Breadth", "Nominator Diversity",
                      "Nominator Reach", "Concentration (Herfindahl)"]

            # Posterior difference plots
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[i // 2, i % 2]
                comp = result["comparison"].get(metric, {})
                idata = result.get("bayesian_results", {}).get(metric)

                if idata is not None:
                    diff_samples = idata.posterior["diff"].values.flatten()
                    ax.hist(diff_samples, bins=50, density=True, alpha=0.7, color="#1f77b4")
                    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No difference")
                    hdi_lo = comp.get("hdi_low", 0)
                    hdi_hi = comp.get("hdi_high", 0)
                    ax.axvline(hdi_lo, color="black", linestyle=":", alpha=0.7)
                    ax.axvline(hdi_hi, color="black", linestyle=":", alpha=0.7)
                    ax.fill_betweenx([0, ax.get_ylim()[1] * 0.1], hdi_lo, hdi_hi,
                                     alpha=0.2, color="black", label="94% HDI")
                    p_greater = comp.get("p_greater", 0.5)
                    ax.set_title(f"{title}\nP(winners > near-misses) = {p_greater:.2f}", fontsize=10)
                    ax.set_xlabel("Difference (winners - near-misses)")
                    ax.legend(fontsize=7)
                else:
                    ax.set_title(f"{title}\n(insufficient data)")

            fig.tight_layout()
            st.pyplot(fig)
            _fig_download_buttons(fig, "near_miss_bayesian", "near_miss_plot")
            plt.close(fig)

            # Summary table
            st.markdown("#### Bayesian Comparison Summary")
            comp_rows = []
            for metric in metrics:
                comp = result["comparison"][metric]
                comp_rows.append({
                    "Metric": metric.replace("_", " ").title(),
                    "Winner Mean": f"{comp.get('mu1_mean', 0):.3f}",
                    "Near-Miss Mean": f"{comp.get('mu2_mean', 0):.3f}",
                    "Diff Mean": f"{comp.get('diff_mean', 0):.3f}",
                    "94% HDI": f"[{comp.get('hdi_low', 0):.3f}, {comp.get('hdi_high', 0):.3f}]",
                    "P(winners>)": f"{comp.get('p_greater', 0.5):.3f}",
                })
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True)

            # Diagnostics per metric
            for metric in metrics:
                idata = result.get("bayesian_results", {}).get(metric)
                if idata is not None:
                    with st.expander(f"Diagnostics: {metric}"):
                        render_bayesian_diagnostics(idata, var_names=["diff"],
                                                     key_prefix=f"nm_{metric}")

            # Top near-misses table
            st.markdown("#### Top 20 Near-Misses by Nomination Count")
            st.dataframe(result["top_near_misses"], hide_index=True)
            dl1, dl2 = st.columns(2)
            with dl1:
                _csv_download_button(result["near_miss_table"], "near_misses.csv", key="near_miss_csv")
            with dl2:
                _csv_download_button(result["winner_table"], "comparable_winners.csv", key="winners_csv")
        else:
            st.info("Click **Run Near-Miss Analysis** in the sidebar.")

    # --- Campaign Success Rate (BAYESIAN) ---
    if flags.get("show_campaign_success"):
        st.subheader("Campaign Success Rate (Bayesian)")
        with st.expander("About this analysis"):
            st.markdown(
                "Tests whether nominees who received coordinated nomination "
                "campaigns win at higher rates than individually-nominated controls.\n\n"
                "**Why Bayesian?** Frequentist tests (Fisher's exact, Cochran-Mantel-Haenszel) "
                "give p-values but cannot directly answer *'How much more likely is a campaign "
                "nominee to win?'*. The Bayesian approach provides:\n"
                "- **Beta-Binomial model:** Estimates the actual win probability for each group "
                "with full uncertainty. You see the posterior distributions of P(win | campaign) "
                "and P(win | control) and their difference — not just 'significant' or 'not "
                "significant'.\n"
                "- **Hierarchical logistic regression:** Controls for nomination count by "
                "giving each stratum its own baseline (random intercept), while sharing "
                "information across strata. This is more principled than the CMH test's "
                "fixed-strata assumption, especially when some strata have very few "
                "observations — the hierarchical model *partially pools* toward the overall "
                "mean rather than discarding sparse strata.")
        if flags.get("run_campaign_success"):
            cs_min = flags.get("cs_min_noms", 5)
            cs_win = flags.get("cs_window", 3)
            with st.spinner("Running Bayesian campaign success analysis..."):
                result = compute_campaign_success(cat_combined, precomputed,
                                                   min_nominations=cs_min,
                                                   year_window=cs_win, **sample_kwargs)
            if "error" in result:
                st.error(result["error"])
            else:
                n_raw = result.get("n_raw_campaigns", 0)
                n_rel = result.get("n_relative_campaigns", 0)
                if n_raw > n_rel:
                    st.caption(f"Relative-burst filter: {n_raw} raw -> {n_rel} anomalous.")

                # Key metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Campaign nominees", result["n_campaign"])
                col2.metric("Campaign win rate", f"{result['campaign_win_rate'] * 100:.1f}%")
                col3.metric("Control win rate", f"{result['control_win_rate'] * 100:.1f}%")

                # Beta-Binomial posterior
                bb = result.get("bb_summary", {})
                if bb:
                    st.markdown("#### Beta-Binomial: Campaign vs Control Win Probability")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("P(campaign)", f"{bb.get('p1_mean', 0):.3f}")
                    col2.metric("P(control)", f"{bb.get('p2_mean', 0):.3f}")
                    col3.metric("Diff", f"{bb.get('diff_mean', 0):.3f}",
                                help=f"94% HDI: [{bb.get('hdi_low', 0):.3f}, {bb.get('hdi_high', 0):.3f}]")
                    col4.metric("P(campaign > control)", f"{bb.get('p_greater', 0.5):.3f}")

                    bb_idata = result.get("bb_idata")
                    if bb_idata is not None:
                        fig_bb, ax_bb = plt.subplots(figsize=(6, 3))
                        diff_samples = bb_idata.posterior["diff"].values.flatten()
                        ax_bb.hist(diff_samples, bins=50, density=True, alpha=0.7, color="#d62728")
                        ax_bb.axvline(0, color="black", linestyle="--")
                        ax_bb.set_xlabel("P(campaign) - P(control)")
                        ax_bb.set_title("Posterior: Difference in Win Probability")
                        fig_bb.tight_layout()
                        st.pyplot(fig_bb)
                        plt.close(fig_bb)
                        render_bayesian_diagnostics(bb_idata, var_names=["diff"],
                                                     key_prefix="cs_bb")

                # Hierarchical logistic
                hl = result.get("hl_summary", {})
                if hl and "error" not in hl:
                    st.markdown("#### Hierarchical Logistic: Group Effect (stratified)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Group effect (beta)", f"{hl.get('beta_mean', 0):.3f}",
                                help=f"94% HDI: [{hl.get('hdi_low', 0):.3f}, {hl.get('hdi_high', 0):.3f}]")
                    col2.metric("Odds ratio", f"{hl.get('odds_ratio_mean', 1):.2f}")
                    col3.metric("P(effect > 0)", f"{hl.get('p_positive', 0.5):.3f}")

                    hl_idata = result.get("hl_idata")
                    if hl_idata is not None:
                        render_bayesian_diagnostics(hl_idata, var_names=["beta_group"],
                                                     key_prefix="cs_hl")

                # Conjugate BF10
                conjugate = result.get("conjugate")
                if conjugate is not None:
                    _render_bayes_factor(conjugate,
                                         title="Bayes Factor: Campaign vs Control (Conjugate)",
                                         key_prefix="cs_bf")

                st.caption(
                    "Controls are matched on both nomination count (+/-30%) and era "
                    "(first nomination year +/-15 years), reducing confounds from "
                    "both fame and time-period effects."
                )

                # Bin breakdown table
                bin_df = result["bin_breakdown"]
                if not bin_df.empty:
                    st.markdown("#### Win Rate by Nomination Count Band")
                    display_rows = []
                    for _, r in bin_df.iterrows():
                        display_rows.append({
                            "Nominations": r["nom_range"],
                            "Campaign N": int(r["campaign_n"]),
                            "Campaign won": int(r["campaign_won"]),
                            "Campaign rate": f"{r['campaign_rate'] * 100:.1f}%",
                            "Control N": int(r["control_n"]),
                            "Control won": int(r["control_won"]),
                            "Control rate": f"{r['control_rate'] * 100:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(display_rows), hide_index=True)

                # Campaign nominees table
                st.markdown("#### Campaign Nominees")
                ct = result["campaign_table"]
                display_cols = ["name", "total_noms", "won", "campaign_noms",
                                "campaign_years", "baseline_rate", "burst_rate"]
                st.dataframe(ct[display_cols], hide_index=True)
                dl1, dl2 = st.columns(2)
                with dl1:
                    _csv_download_button(ct, "campaign_nominees.csv", key="campaign_success_csv")
                with dl2:
                    _csv_download_button(result["control_table"], "matched_controls.csv", key="control_csv")
        else:
            st.info("Click **Run Campaign Success Analysis** in the sidebar.")

    # --- Burst Decomposition (BAYESIAN) ---
    if flags.get("show_burst_decomp"):
        st.subheader("Burst Decomposition: Insiders vs Outsiders (Bayesian)")
        with st.expander("About this analysis"):
            st.markdown(
                "Decomposes nomination bursts into *insider* (connected to existing "
                "nomination networks) and *outsider* (independent) patterns, and tests "
                "whether network-connected nominees win more often.\n\n"
                "**Why Bayesian?** Three frequentist tests are replaced:\n"
                "- **Beta-Binomial** (replaces Fisher's exact): Gives the full posterior "
                "of the win-rate difference between insiders and outsiders, with calibrated "
                "uncertainty — not just a p-value.\n"
                "- **Hierarchical logistic** (replaces CMH): Shares information across "
                "nomination-count strata via partial pooling. When a stratum has only 2-3 "
                "observations, the frequentist CMH either ignores it or gives it equal weight. "
                "The hierarchical model shrinks small-sample strata toward the grand mean, "
                "producing more stable estimates.\n"
                "- **Bayesian logistic regression** (replaces point-biserial correlation): "
                "Directly models how country diversity predicts winning on the probability "
                "scale, with a full posterior on the effect size. Point-biserial correlation "
                "assumes a linear relationship with a continuous outcome; Bayesian logistic "
                "correctly models the binary win/loss outcome.")
        if flags.get("run_burst_decomp"):
            bd_min = flags.get("bd_min_noms", 5)
            bd_win = flags.get("bd_window", 3)
            with st.spinner("Running Bayesian burst decomposition..."):
                result = compute_burst_decomposition(cat_combined, precomputed,
                                                      min_nominations=bd_min,
                                                      year_window=bd_win, **sample_kwargs)
            if "error" in result:
                st.error(result["error"])
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Outsiders (independent)", result["n_independent"])
                col2.metric("Insiders (connected)", result["n_connected"])
                col3.metric("Outsider win rate", f"{result['independent_win_rate'] * 100:.1f}%")
                col4.metric("Insider win rate", f"{result['connected_win_rate'] * 100:.1f}%")

                # Beta-Binomial
                bb = result.get("bb_summary", {})
                if bb:
                    st.markdown("#### Beta-Binomial: Connected vs Independent Win Probability")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Diff (connected - independent)",
                                f"{bb.get('diff_mean', 0):.3f}",
                                help=f"94% HDI: [{bb.get('hdi_low', 0):.3f}, {bb.get('hdi_high', 0):.3f}]")
                    col2.metric("Odds ratio", f"{bb.get('odds_ratio_mean', 1):.2f}")
                    col3.metric("P(connected > independent)", f"{bb.get('p_greater', 0.5):.3f}")

                    bb_idata = result.get("bb_idata")
                    if bb_idata is not None:
                        render_bayesian_diagnostics(bb_idata, var_names=["diff"],
                                                     key_prefix="bd_bb")

                # Hierarchical logistic
                hl = result.get("hl_summary", {})
                if hl and "error" not in hl:
                    st.markdown("#### Hierarchical Logistic: Group Effect (stratified)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Group effect (beta)", f"{hl.get('beta_mean', 0):.3f}")
                    col2.metric("Odds ratio", f"{hl.get('odds_ratio_mean', 1):.2f}")
                    col3.metric("P(effect > 0)", f"{hl.get('p_positive', 0.5):.3f}")

                    hl_idata = result.get("hl_idata")
                    if hl_idata is not None:
                        render_bayesian_diagnostics(hl_idata, var_names=["beta_group"],
                                                     key_prefix="bd_hl")

                # Bayesian logistic: n_countries
                bl = result.get("bl_summary", {})
                if bl and "error" not in bl:
                    st.markdown("#### Country Diversity vs Winning (Bayesian Logistic)")
                    col1, col2 = st.columns(2)
                    col1.metric(f"Beta (n_countries)", f"{bl.get('beta_mean', 0):.3f}",
                                help=f"94% HDI: [{bl.get('hdi_low', 0):.3f}, {bl.get('hdi_high', 0):.3f}]")
                    col2.metric("P(positive)", f"{bl.get('p_positive', 0.5):.3f}")

                    bl_idata = result.get("bl_idata")
                    if bl_idata is not None:
                        render_bayesian_diagnostics(bl_idata,
                                                     var_names=["beta_n_countries"],
                                                     key_prefix="bd_bl")

                # Conjugate BF10
                conjugate = result.get("conjugate")
                if conjugate is not None:
                    _render_bayes_factor(conjugate,
                                         title="Bayes Factor: Connected vs Independent (Conjugate)",
                                         key_prefix="bd_bf")

                # Dose-response logistic
                dr_summary = result.get("dr_summary", {})
                if dr_summary and "coefficients" in dr_summary:
                    st.markdown("#### Dose-Response: Connected Fraction + Nomination Volume")
                    st.caption(
                        "Logistic regression with two predictors: **connected_frac** "
                        "(what fraction of burst nominators are co-active) and "
                        "**log(1 + total_noms)** (nomination volume). Tests whether "
                        "the insider effect persists after controlling for fame."
                    )
                    dr_coefs = dr_summary["coefficients"]
                    dr_cols = st.columns(len(dr_coefs))
                    for i, (fname, fc) in enumerate(dr_coefs.items()):
                        dr_cols[i].metric(
                            f"Beta ({fname})",
                            f"{fc['mean']:.3f}",
                            help=f"94% HDI: [{fc['hdi_low']:.3f}, {fc['hdi_high']:.3f}], "
                                 f"P(>0) = {fc['p_positive']:.3f}")
                    dr_idata = result.get("dr_idata")
                    if dr_idata is not None:
                        render_bayesian_diagnostics(dr_idata, var_names=["beta"],
                                                     key_prefix="bd_dr")

                # Science-only subset
                science_only = result.get("science_only")
                if science_only is not None:
                    st.markdown("#### Science Categories Only (Physics, Chemistry, Medicine)")
                    st.caption(
                        "Restricts to science Nobel categories to test whether the "
                        "insider/outsider effect holds in disciplines with clearer "
                        "empirical achievement criteria."
                    )
                    _render_bayes_factor(science_only,
                                         title="Science-Only: Connected vs Independent",
                                         key_prefix="bd_sci")

                # Per-category breakdown
                per_category = result.get("per_category", {})
                if per_category:
                    st.markdown("#### Per-Category Breakdown")
                    st.caption(
                        "Conjugate BF10 for each category with at least 4 nominees "
                        "per group. BF10 > 3 suggests moderate evidence for different "
                        "win rates between insiders and outsiders."
                    )
                    cat_rows = []
                    for cat, cat_bf in sorted(per_category.items()):
                        cat_rows.append({
                            "Category": cat,
                            "Connected N": cat_bf["n1"],
                            "Connected won": cat_bf["k1"],
                            "Independent N": cat_bf["n2"],
                            "Independent won": cat_bf["k2"],
                            "Connected rate": f"{cat_bf['p1_mean']:.3f}",
                            "Independent rate": f"{cat_bf['p2_mean']:.3f}",
                            "BF10": f"{cat_bf['bf10']:.2f}",
                            "Evidence": cat_bf["bf10_interpretation"],
                        })
                    st.dataframe(pd.DataFrame(cat_rows), hide_index=True)

                # Bin breakdown
                bin_df = result["bin_breakdown"]
                if not bin_df.empty:
                    st.markdown("#### Win Rate by Nomination Count Band")
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    x_pos = np.arange(len(bin_df))
                    bar_w = 0.35
                    ax.bar(x_pos - bar_w / 2, bin_df["insider_rate"].values * 100,
                           bar_w, color="#d62728", edgecolor="black", label="Insider (connected)")
                    ax.bar(x_pos + bar_w / 2, bin_df["outsider_rate"].values * 100,
                           bar_w, color="#2ca02c", edgecolor="black", label="Outsider (independent)")
                    tick_labels = [
                        f"{r['nom_range']}\n(I:{int(r['insider_n'])}, O:{int(r['outsider_n'])})"
                        for _, r in bin_df.iterrows()]
                    ax.set_xticks(x_pos); ax.set_xticklabels(tick_labels, fontsize=8)
                    ax.set_xlabel("Total lifetime nominations")
                    ax.set_ylabel("Win rate (%)")
                    ax.set_title("Insider vs. Outsider Win Rate by Nomination Count")
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig)
                    _fig_download_buttons(fig, "burst_decomposition", "burst_decomp")
                    plt.close(fig)

                # Burst table
                st.markdown("#### Top Burst Nominees")
                bt = result["burst_table"]
                display_cols = [c for c in ["nominee", "won", "total_noms", "burst_noms",
                                "n_burst_nominators", "diversity", "density",
                                "connected_frac", "n_countries", "burst_type"] if c in bt.columns]
                st.dataframe(bt[display_cols].head(30), hide_index=True)
                _csv_download_button(bt, "burst_decomposition.csv", key="burst_decomp_csv")
        else:
            st.info("Click **Run Burst Decomposition** in the sidebar.")

    # --- Centrality Predicts Winners (BAYESIAN) ---
    if flags.get("show_centrality"):
        st.subheader("Centrality Predicts Winners (Bayesian)")
        with st.expander("About this analysis"):
            st.markdown(
                "Fits four logistic regression models to predict Nobel winners from "
                "network centrality features, then compares their predictive power.\n\n"
                "**Why Bayesian?** Frequentist logistic regression (sklearn) gives point "
                "estimates of coefficients and a single ROC curve. The Bayesian version "
                "provides:\n"
                "- **Uncertainty on every coefficient:** Full posteriors show which "
                "predictors are reliably important vs. poorly constrained by the data.\n"
                "- **Principled model comparison:** WAIC and LOO-CV (leave-one-out "
                "cross-validation via Pareto-smoothed importance sampling) estimate "
                "out-of-sample predictive accuracy without actually splitting the data. "
                "This is more efficient than train/test splits and more informative than "
                "AIC/BIC.\n"
                "- **ROC curves with credible bands:** Instead of a single ROC line, each "
                "posterior draw produces a different ROC curve, giving a shaded 94% credible "
                "band. This shows how much the ROC varies given parameter uncertainty — wide "
                "bands mean the apparent predictive power is fragile.\n"
                "- **Regularization via priors:** The Normal(0, 1) priors on standardized "
                "coefficients act as gentle regularization, reducing overfitting compared to "
                "unpenalized frequentist logistic regression.")
        if flags.get("run_centrality"):
            with st.spinner("Running Bayesian centrality analysis (4 PyMC models)..."):
                result = compute_centrality_prediction(cat_combined, precomputed, **sample_kwargs)
            if "error" in result:
                st.error(result["error"])
            else:
                model_results = result["model_results"]
                roc_data = result["roc_data"]

                # AUC summary
                st.markdown("#### Model Comparison: Posterior Predictive AUC")
                model_display = [
                    ("in_degree_only", "In-degree only"),
                    ("structural", "Structural"),
                    ("full", "Full"),
                    ("full_nompr", "Full + Nom.PR"),
                ]
                auc_cols = st.columns(len(model_display))
                for i, (name, label) in enumerate(model_display):
                    roc = roc_data.get(name, {})
                    if "error" not in roc and roc:
                        auc_cols[i].metric(
                            f"AUC ({label})",
                            f"{roc.get('auc_mean', 0):.3f}",
                            help=f"94% HDI: [{roc.get('auc_hdi_low', 0):.3f}, {roc.get('auc_hdi_high', 0):.3f}]")

                # WAIC/LOO comparison table
                comparison_df = result.get("comparison_df")
                if comparison_df is not None:
                    st.markdown("#### WAIC/LOO Model Comparison")
                    st.dataframe(comparison_df, use_container_width=True)

                # Posterior predictive ROC with credible bands
                fig, ax = plt.subplots(figsize=(7, 6))
                colors = {"in_degree_only": "#999999", "structural": "#d62728",
                          "full": "#1f77b4", "full_nompr": "#ff7f0e"}
                labels = {"in_degree_only": "In-degree only", "structural": "Structural",
                          "full": "Full", "full_nompr": "Full + Nom.PR"}

                for name in ["in_degree_only", "structural", "full", "full_nompr"]:
                    roc = roc_data.get(name, {})
                    if "error" in roc or "mean_fpr" not in roc:
                        continue
                    auc_mean = roc["auc_mean"]
                    ax.plot(roc["mean_fpr"], roc["mean_tpr"],
                            color=colors[name], linewidth=2,
                            label=f"{labels[name]} (AUC={auc_mean:.3f})")
                    ax.fill_between(roc["mean_fpr"], roc["tpr_low"], roc["tpr_high"],
                                    color=colors[name], alpha=0.15)

                ax.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.3)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Posterior Predictive ROC Curves (94% Credible Bands)")
                ax.legend(loc="lower right")
                fig.tight_layout()
                st.pyplot(fig)
                _fig_download_buttons(fig, "centrality_roc_bayesian", "centrality_roc")
                plt.close(fig)

                # AUC posterior distributions
                fig_auc, ax_auc = plt.subplots(figsize=(7, 3))
                for name in ["in_degree_only", "structural", "full", "full_nompr"]:
                    roc = roc_data.get(name, {})
                    if "auc_samples" in roc:
                        ax_auc.hist(roc["auc_samples"], bins=40, alpha=0.4,
                                    color=colors.get(name, "#333"),
                                    label=labels.get(name, name), density=True)
                ax_auc.set_xlabel("AUC")
                ax_auc.set_title("Posterior AUC Distributions")
                ax_auc.legend()
                fig_auc.tight_layout()
                st.pyplot(fig_auc)
                plt.close(fig_auc)

                # Coefficient forest plot (Full + Nom.PR model if available, else Full)
                best_model = "full_nompr" if "full_nompr" in model_results else "full"
                best_label = labels.get(best_model, best_model)
                st.markdown(f"#### Coefficient Posteriors ({best_label} Model)")
                best_coefs = model_results.get(best_model, {}).get("coefficients", {})
                if best_coefs:
                    fig_coef, ax_coef = plt.subplots(figsize=(7, max(3.5, len(best_coefs) * 0.5)))
                    names = list(best_coefs.keys())
                    means = [best_coefs[n]["mean"] for n in names]
                    lows = [best_coefs[n]["hdi_low"] for n in names]
                    highs = [best_coefs[n]["hdi_high"] for n in names]
                    y_pos = range(len(names))
                    ax_coef.barh(y_pos, means, xerr=[[m - l for m, l in zip(means, lows)],
                                                       [h - m for m, h in zip(means, highs)]],
                                 color=colors.get(best_model, "#1f77b4"), alpha=0.7, capsize=4)
                    ax_coef.set_yticks(y_pos)
                    ax_coef.set_yticklabels(names)
                    ax_coef.axvline(0, color="red", linestyle="--", alpha=0.5)
                    ax_coef.set_xlabel("Coefficient (standardized)")
                    ax_coef.set_title(f"{best_label} Model: Feature Coefficients with 94% HDI")
                    fig_coef.tight_layout()
                    st.pyplot(fig_coef)
                    _fig_download_buttons(fig_coef, "centrality_coefficients", "centrality_coef")
                    plt.close(fig_coef)

                    # Coefficient table
                    coef_rows = []
                    for name in best_coefs:
                        c = best_coefs[name]
                        coef_rows.append({
                            "Feature": name,
                            "Mean": f"{c['mean']:.4f}",
                            "94% HDI": f"[{c['hdi_low']:.4f}, {c['hdi_high']:.4f}]",
                            "P(positive)": f"{c['p_positive']:.3f}",
                        })
                    st.dataframe(pd.DataFrame(coef_rows), hide_index=True)

                # Diagnostics for each model
                idatas = result.get("idatas", {})
                for model_name in ["in_degree_only", "structural", "full", "full_nompr"]:
                    idata = idatas.get(model_name)
                    if idata is not None:
                        with st.expander(f"Diagnostics: {labels.get(model_name, model_name)}"):
                            render_bayesian_diagnostics(idata, var_names=["alpha", "beta"],
                                                         key_prefix=f"cent_{model_name}")

                st.caption("Coefficients are on standardized features.")
                _csv_download_button(result["feature_table"], "centrality_features.csv",
                                     key="centrality_csv")
        else:
            st.info("Click **Run Centrality Analysis** in the sidebar.")
