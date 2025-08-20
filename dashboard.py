import json
import io
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="Schema Mapping Dashboard", layout="wide")

st.sidebar.title("Inputs")
default_csv = "outputs/mapping_suggestions.csv"
default_json = "outputs/mapping_suggestions.json"

csv_file = st.sidebar.file_uploader("Upload mapping_suggestions.csv", type=["csv"])
json_file = st.sidebar.file_uploader("Upload mapping_suggestions.json", type=["json"])


if csv_file is None and Path(default_csv).exists():
    csv_path = default_csv
    df = pd.read_csv(csv_path)
else:
    if csv_file is None:
        st.warning("Please upload mapping_suggestions.csv or place it at outputs/mapping_suggestions.csv")
        st.stop()
    df = pd.read_csv(csv_file)

if json_file is None and Path(default_json).exists():
    with open(default_json, "r", encoding="utf-8") as f:
        details = json.load(f)
else:
    if json_file is None:
        details = None
    else:
        details = json.load(json_file)


expected_cols = {
    "source_table": ["source_table", "src_table", "sourceTable"],
    "source_column": ["source_column", "src_column", "sourceColumn"],
    "target_table": ["target_table", "tgt_table", "targetTable"],
    "predicted_target_column": ["predicted_target_column", "target_column", "predictedTargetColumn", "prediction"],
    "score": ["score", "probability", "confidence"]
}

col_map = {}
for canon, alts in expected_cols.items():
    for c in df.columns:
        if c in alts:
            col_map[c] = canon
            break
df = df.rename(columns=col_map)

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error(f"Missing expected columns in CSV: {missing}. Please check your pipeline output headers.")
    st.stop()


tables_source = sorted(df["source_table"].dropna().unique().tolist())
tables_target = sorted(df["target_table"].dropna().unique().tolist())

st.sidebar.markdown("---")
thresh = st.sidebar.slider("Confidence threshold (matched if score ≥ threshold)", 0.0, 1.0, 0.70, 0.01)
src_filter = st.sidebar.multiselect("Filter by source table(s)", tables_source, default=tables_source)
tgt_filter = st.sidebar.multiselect("Filter by target table(s)", tables_target, default=tables_target)

df = df[df["source_table"].isin(src_filter) & df["target_table"].isin(tgt_filter)].copy()


def build_alt_lookup(js):
    lookup = {}
    if not isinstance(js, list):
        return lookup
    for item in js:
        key = (item.get("source_table"), item.get("source_column"))
        alts = item.get("alternates", [])
        lookup[key] = alts
    return lookup

alts_lookup = build_alt_lookup(details) if details else {}

def stringify_alts(source_table, source_column, top_n=3):
    alts = alts_lookup.get((source_table, source_column), [])
    if not alts:
        return ""
    alts_sorted = sorted(alts, key=lambda x: x.get("score", 0), reverse=True)[:top_n]
    return "; ".join(f"{a.get('t_col','?')} ({a.get('score',0):.2f})" for a in alts_sorted)

df["alternates_top3"] = df.apply(lambda r: stringify_alts(r["source_table"], r["source_column"]), axis=1)


df["is_matched"] = df["predicted_target_column"].notna() & (df["score"] >= thresh)


st.title("Schema Mapping Dashboard")
st.caption("Guidewire → InsureNow | Shows mapping quality, confidence distribution, and review queue")


total_src_cols = len(df)
matched = int(df["is_matched"].sum())
not_matched = total_src_cols - matched
match_pct = (matched / total_src_cols * 100.0) if total_src_cols else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Source Columns (filtered)", total_src_cols)
c2.metric("Matched (score ≥ threshold)", matched)
c3.metric("Not Matched", not_matched)
c4.metric("% Matched", f"{match_pct:.1f}%")


ch1, ch2, ch3 = st.columns([1, 1, 1])


with ch1:
    st.subheader("Matched vs Not Matched")
    fig, ax = plt.subplots()
    vals = [matched, not_matched]
    labels = ["Matched", "Not Matched"]
    ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)


with ch2:
    st.subheader("Match Rate by Source Table")
    by_src = df.groupby("source_table").agg(
        total=("source_column", "count"),
        matched=("is_matched", "sum"),
    ).reset_index()
    by_src["match_rate"] = np.where(by_src["total"] > 0, by_src["matched"] / by_src["total"] * 100.0, 0.0)
    fig2, ax2 = plt.subplots()
    ax2.bar(by_src["source_table"], by_src["match_rate"])
    ax2.set_ylabel("Match Rate (%)")
    ax2.set_xticklabels(by_src["source_table"], rotation=45, ha="right")
    st.pyplot(fig2)


with ch3:
    st.subheader("Confidence Score Distribution")
    fig3, ax3 = plt.subplots()
    df["score"].dropna().plot(kind="hist", bins=12, ax=ax3)
    ax3.axvline(thresh, linestyle="--")
    ax3.set_xlabel("Score")
    st.pyplot(fig3)

st.markdown("---")


left, right = st.columns([1.2, 1])

with left:
    st.subheader("Low-Confidence Mappings (below threshold)")
    low_conf = df[df["score"] < thresh].sort_values("score")
    cols_to_show = [
        "source_table", "source_column", "target_table",
        "predicted_target_column", "score", "alternates_top3"
    ]
    st.dataframe(low_conf[cols_to_show], use_container_width=True)

    
    buf = io.BytesIO()
    low_conf[cols_to_show].to_csv(buf, index=False)
    st.download_button(
        label="Download Low-Confidence Review CSV",
        data=buf.getvalue(),
        file_name="low_confidence_review.csv",
        mime="text/csv"
    )

with right:
    st.subheader("Top Matches (sorted by score)")
    top = df[df["is_matched"]].sort_values("score", ascending=False)
    st.dataframe(top[cols_to_show], use_container_width=True)

    
    buf2 = io.BytesIO()
    top[cols_to_show].to_csv(buf2, index=False)
    st.download_button(
        label="Download Matched CSV",
        data=buf2.getvalue(),
        file_name="matched_columns.csv",
        mime="text/csv"
    )


st.markdown("---")
st.subheader("Matches by Target Table")
by_tgt = df.groupby("target_table").agg(
    total=("source_column", "count"),
    matched=("is_matched", "sum")
).reset_index()
by_tgt["match_rate"] = np.where(by_tgt["total"] > 0, by_tgt["matched"] / by_tgt["total"] * 100.0, 0.0)
st.dataframe(by_tgt.sort_values("match_rate", ascending=False), use_container_width=True)

st.caption(
    "Tip: Use the sidebar to change the confidence threshold and filter by specific tables. "
    "If JSON was provided, the review table includes alternates (top-3) for each source column."
)
