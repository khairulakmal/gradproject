import json
import os
import re
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import warnings
from transformers import pipeline
from wordcloud import WordCloud

from reddit_fetcher import reddit_fetch_ui, parse_timestamp

# -----------------------------------------
# GLOBAL CONFIG
# -----------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="RedditSense - Emotion Analysis",
    layout="wide",
)

if not hasattr(st.session_state, "_initialized"):
    st.session_state.clear()
    st.session_state._initialized = True
    st.session_state.analyzed_data = None
    st.session_state.fetched_data = None

# -----------------------------------------
# CONSTANTS
# -----------------------------------------
NEGATIVE_KEYWORDS = [
    "end production", "job losses", "dispute", "layoff", "fired", "terminated",
    "bankruptcy", "bankrupt", "crisis", "strike", "protest", "lawsuit", "sue",
    "recall", "recalled", "recalling", "recalls"
]

BUSINESS_TERMS = [
    "stock", "market", "price", "share", "profit", "loss", "revenue",
    "earnings", "dividend", "investment", "quarterly", "annual", "growth"
]

# -----------------------------------------
# HELPERS
# -----------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:2000]


def has_negative_keywords(text):
    text = text.lower()
    return any(keyword in text for keyword in NEGATIVE_KEYWORDS)


def has_business_terms(text):
    text = text.lower()
    return any(term in text for term in BUSINESS_TERMS)


def is_valid_dataframe(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    required = ["emotion", "sentiment", "sentiment_score", "full_text"]
    return all(col in df.columns for col in required)


def calculate_emotion_entropy(emotions):
    if not emotions:
        return 0.0
    counts = Counter(emotions)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return float(-sum(p * np.log2(p) for p in probs if p > 0))


def kl_divergence(p, q, eps=1e-9):
    """
    KL divergence D_KL(p || q) with smoothing.
    p, q are 1D arrays representing probability distributions.
    """
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# -----------------------------------------
# EVENT DETECTION
# -----------------------------------------
def detect_events(df_posts):
    """
    Detect:
    - Sentiment spikes (z-score |z| > 2)
    - Activity spikes (post count z > 2)
    - Emotion shifts (daily emotion distribution KL divergence > threshold)
    Returns dict with three DataFrames:
    {
        "sentiment_spikes": ...,
        "activity_spikes": ...,
        "emotion_shifts": ...,
    }
    """

    if df_posts is None or df_posts.empty:
        return {
            "sentiment_spikes": pd.DataFrame(),
            "activity_spikes": pd.DataFrame(),
            "emotion_shifts": pd.DataFrame(),
        }

    df = df_posts.copy()
    df["date"] = df["date"].apply(parse_timestamp)
    df = df.dropna(subset=["date"]).sort_values("date")

    if df.empty:
        return {
            "sentiment_spikes": pd.DataFrame(),
            "activity_spikes": pd.DataFrame(),
            "emotion_shifts": pd.DataFrame(),
        }

    # ---------------- SENTIMENT SPIKES ----------------
    daily_sent = (
        df.resample("D", on="date")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": "sentiment_mean"})
    )

    mu = daily_sent["sentiment_mean"].mean()
    sigma = daily_sent["sentiment_mean"].std()

    if sigma and sigma > 0:
        daily_sent["z"] = (daily_sent["sentiment_mean"] - mu) / sigma
        sent_spikes = daily_sent[daily_sent["z"].abs() > 2].copy()
        sent_spikes["type"] = "Sentiment Spike"
    else:
        sent_spikes = pd.DataFrame(columns=["date", "sentiment_mean", "z", "type"])

    # ---------------- ACTIVITY SPIKES ----------------
    daily_count = (
        df.resample("D", on="date")["id"]
        .count()
        .reset_index()
        .rename(columns={"id": "post_count"})
    )

    mu_c = daily_count["post_count"].mean()
    sigma_c = daily_count["post_count"].std()

    if sigma_c and sigma_c > 0:
        daily_count["z"] = (daily_count["post_count"] - mu_c) / sigma_c
        act_spikes = daily_count[daily_count["z"] > 2].copy()
        act_spikes["type"] = "Activity Spike"
    else:
        act_spikes = pd.DataFrame(columns=["date", "post_count", "z", "type"])

    # ---------------- EMOTION SHIFT EVENTS ----------------
    emo_daily = (
        df.groupby([pd.Grouper(key="date", freq="D"), "emotion"])
        .size()
        .unstack(fill_value=0)
    )

    if emo_daily.empty or emo_daily.shape[0] < 2:
        emo_shift_events = pd.DataFrame(columns=["date", "kl_divergence", "type"])
    else:
        emo_shift_events_list = []
        prev_row = None
        prev_index = None
        for idx, row in emo_daily.iterrows():
            if prev_row is not None:
                kl = kl_divergence(prev_row.values, row.values)
                # Threshold tuned by feel; you can change if too sensitive.
                if kl > 0.35:
                    emo_shift_events_list.append(
                        {
                            "date": idx,
                            "kl_divergence": float(kl),
                            "type": "Emotion Shift",
                        }
                    )
            prev_row = row
            prev_index = idx
        emo_shift_events = pd.DataFrame(emo_shift_events_list)

    return {
        "sentiment_spikes": sent_spikes,
        "activity_spikes": act_spikes,
        "emotion_shifts": emo_shift_events,
    }


# -----------------------------------------
# MODEL LOADERS
# -----------------------------------------
@st.cache_resource(show_spinner="Loading emotion model...")
def load_emotion_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-emotion-latest",
        top_k=1,
        device=device,
    )


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )


# -----------------------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------------------
def process_data(data, emotion_pipe, sentiment_pipe, progress=None):

    required_cols = [
        "id", "subreddit", "text", "full_text", "emotion", "emotion_score",
        "sentiment", "sentiment_score", "is_confident", "needs_review",
        "date", "url", "score"
    ]

    if data is None or (isinstance(data, (list, dict, pd.DataFrame)) and len(data) == 0):
        return pd.DataFrame(columns=required_cols)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    df["title"] = df.get("title", "")
    df["text"] = df.get("text", df.get("selftext", ""))
    df["created_utc"] = df.get("created_utc", datetime.now())
    df["created_utc"] = df["created_utc"].apply(parse_timestamp)

    results = []
    total = len(df)

    for i, row in df.iterrows():
        try:
            txt = clean_text(f"{row.get('title', '')} {row.get('text', '')}")
            if not txt:
                continue

            emo_res = emotion_pipe(txt)[0][0]
            emotion = emo_res["label"].lower()
            emotion_score = float(emo_res["score"])

            sent_res = sentiment_pipe(txt)[0]
            sentiment = sent_res["label"].upper()
            sentiment_score = float(sent_res["score"])
            if sentiment == "NEGATIVE":
                sentiment_score = -sentiment_score
            elif sentiment == "NEUTRAL":
                sentiment_score = 0.0

            needs_review = False
            if has_negative_keywords(txt) and emotion == "joy":
                emotion = "sadness"
                needs_review = True
            if abs(sentiment_score) > 0.7 and has_business_terms(txt):
                needs_review = True

            results.append({
                "id": row.get("id"),
                "subreddit": row.get("subreddit"),
                "text": txt[:500] + ("..." if len(txt) > 500 else ""),
                "full_text": txt,
                "emotion": emotion,
                "emotion_score": emotion_score,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "is_confident": emotion_score > 0.7,
                "needs_review": needs_review,
                "date": row.get("created_utc"),
                "url": row.get("url"),
                "score": row.get("score", 0),
            })

            if progress:
                progress.progress((i + 1) / total)

        except Exception as e:
            st.warning(f"Error processing row {i}: {e}")
            continue

    out = pd.DataFrame(results)

    if out.empty:
        return pd.DataFrame(columns=required_cols)

    out["date"] = out["date"].apply(parse_timestamp)
    out["emotion_score"] = pd.to_numeric(out["emotion_score"], errors="coerce").fillna(0)
    out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce").fillna(0)
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0)

    return out[required_cols]


# -----------------------------------------
# WORD CLOUD
# -----------------------------------------
def generate_wordcloud(texts, title):
    text = " ".join([t for t in texts if isinstance(t, str)])
    if not text.strip():
        return None
    wc = WordCloud(width=1000, height=500, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    return fig


# -----------------------------------------
# MAIN APP
# -----------------------------------------
def main():
    st.title("RedditSense - Emotion & Sentiment Analysis")

    tab_fetch, tab_analyze = st.tabs(["Fetch Data", "Analyze Data"])

    # ------------------------ FETCH TAB ------------------------
    with tab_fetch:
        st.header("Fetch Data")
        df_fetched = reddit_fetch_ui()

        if isinstance(df_fetched, pd.DataFrame) and not df_fetched.empty:
            st.session_state.fetched_data = df_fetched
            st.success(f"Fetched {len(df_fetched)} posts")
            st.dataframe(df_fetched.head(), use_container_width=True)

    # ------------------------ ANALYZE TAB ------------------------
    with tab_analyze:
        st.header("Analyze Data")

        # Analyze fetched data button
        if (
            "fetched_data" in st.session_state
            and isinstance(st.session_state.fetched_data, pd.DataFrame)
            and not st.session_state.fetched_data.empty
        ):
            if st.button("Analyze Fetched Data", type="primary"):
                emo = load_emotion_model()
                sent = load_sentiment_model()
                pb = st.progress(0.0)
                st.session_state.analyzed_data = process_data(
                    st.session_state.fetched_data, emo, sent, pb
                )
                pb.empty()
                st.success("Analysis of fetched data complete")

        # Upload file and analyze
        uploaded_file = st.file_uploader(
            "Or upload a JSON/CSV file for analysis", type=["json", "csv"]
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".json"):
                    raw = json.load(uploaded_file)
                    if not isinstance(raw, list):
                        st.error("JSON must be a list of objects")
                        raw = []
                else:
                    df_up = pd.read_csv(uploaded_file)
                    raw = df_up.to_dict("records")

                if st.button("Analyze Uploaded Data", type="secondary"):
                    emo = load_emotion_model()
                    sent = load_sentiment_model()
                    pb = st.progress(0.0)
                    st.session_state.analyzed_data = process_data(raw, emo, sent, pb)
                    pb.empty()
                    st.success("Analysis of uploaded data complete")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
                return

        # If we have analyzed data, show dashboards
        if not is_valid_dataframe(st.session_state.get("analyzed_data")):
            st.info("Please fetch or upload data and run analysis.")
            return

        results_df = st.session_state.analyzed_data

        st.subheader("Analyzed Data Preview")
        st.dataframe(results_df.head(), use_container_width=True)

        # ------------------ SUBREDDIT MULTI-SELECT ------------------
        st.markdown("### Subreddit Selection and Comparison")

        all_subs = sorted(results_df["subreddit"].unique())
        selected_subreddits = st.multiselect(
            "Select subreddits to include in analysis",
            options=all_subs,
            default=all_subs,
            help="You can compare multiple subreddits at once.",
        )

        if selected_subreddits:
            filtered_df = results_df[results_df["subreddit"].isin(selected_subreddits)]
        else:
            filtered_df = results_df.iloc[0:0]

        st.write(
            f"Showing {len(filtered_df)} posts from "
            f"{len(selected_subreddits) if selected_subreddits else 0} subreddit(s)."
        )

        # Prepare df_tl and events once, to reuse in Timeline + Event Dashboard
        if filtered_df.empty:
            df_tl = pd.DataFrame()
            events = {
                "sentiment_spikes": pd.DataFrame(),
                "activity_spikes": pd.DataFrame(),
                "emotion_shifts": pd.DataFrame(),
            }
        else:
            df_tl = filtered_df.copy()
            df_tl["date"] = df_tl["date"].apply(parse_timestamp)
            df_tl = df_tl.dropna(subset=["date"]).sort_values("date")
            events = detect_events(df_tl)

        # ------------------ ANALYSIS TABS ----------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Emotion Analysis", "Timeline", "Word Cloud", "Review Needed", "Event Dashboard"]
        )

        # ---------- Emotion Analysis ----------
        with tab1:
            st.subheader("Emotion Distribution")

            if filtered_df.empty:
                st.warning("No data available for selected subreddits.")
            else:
                emo_counts = filtered_df["emotion"].value_counts().reset_index()
                emo_counts.columns = ["Emotion", "Count"]

                fig = px.bar(
                    emo_counts,
                    x="Emotion",
                    y="Count",
                    color="Emotion",
                    title="Emotion Distribution (Selected Subreddits)",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Average Sentiment by Emotion")
                sent_avg = (
                    filtered_df.groupby("emotion")["sentiment_score"]
                    .mean()
                    .sort_values()
                )
                fig2 = px.bar(
                    x=sent_avg.index,
                    y=sent_avg.values,
                    color=sent_avg.values,
                    color_continuous_scale="RdYlGn",
                    title="Average Sentiment by Emotion (Selected Subreddits)",
                    text_auto=".2f",
                )
                fig2.update_layout(coloraxis_showscale=False)
                fig2.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig2, use_container_width=True)

                # Heatmap: Emotion x Subreddit
                st.subheader("Emotion × Subreddit Heatmap")
                heat_data = (
                    filtered_df
                    .pivot_table(index="emotion", columns="subreddit", values="id",
                                 aggfunc="count", fill_value=0)
                )
                if not heat_data.empty:
                    fig_hm = px.imshow(
                        heat_data,
                        labels=dict(x="Subreddit", y="Emotion", color="Post Count"),
                        x=heat_data.columns,
                        y=heat_data.index,
                        title="Emotion vs Subreddit (Post Counts)",
                        aspect="auto",
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    st.warning("Not enough data to build heatmap.")

                # Ranking tables
                st.subheader("Subreddit Rankings")

                sub_stats = (
                    filtered_df.groupby("subreddit")
                    .agg(
                        avg_sentiment=("sentiment_score", "mean"),
                        sentiment_std=("sentiment_score", "std"),
                        num_posts=("id", "count"),
                    )
                    .reset_index()
                )

                # Emotion entropy per subreddit
                entropy_rows = []
                temp = filtered_df.copy()
                temp["date"] = temp["date"].apply(parse_timestamp)
                temp = temp.dropna(subset=["date"])
                if not temp.empty:
                    for sub, sub_df in temp.groupby("subreddit"):
                        per_day = (
                            sub_df.groupby(pd.Grouper(key="date", freq="D"))["emotion"]
                            .apply(list)
                            .reset_index()
                        )
                        per_day["entropy"] = per_day["emotion"].apply(
                            calculate_emotion_entropy
                        )
                        avg_entropy = per_day["entropy"].mean() if not per_day.empty else 0.0
                        entropy_rows.append(
                            {"subreddit": sub, "avg_emotion_entropy": avg_entropy}
                        )

                entropy_df = pd.DataFrame(entropy_rows)
                sub_stats = sub_stats.merge(entropy_df, on="subreddit", how="left")
                sub_stats["sentiment_std"] = sub_stats["sentiment_std"].fillna(0.0)
                sub_stats["avg_emotion_entropy"] = sub_stats[
                    "avg_emotion_entropy"
                ].fillna(0.0)

                st.markdown("**Most Positive Subreddits (by average sentiment)**")
                st.dataframe(
                    sub_stats.sort_values("avg_sentiment", ascending=False),
                    use_container_width=True,
                )

                st.markdown(
                    "**Most Volatile Subreddits (by sentiment std and emotion entropy)**"
                )
                volatile_sorted = sub_stats.sort_values(
                    ["sentiment_std", "avg_emotion_entropy"], ascending=False
                )
                st.dataframe(volatile_sorted, use_container_width=True)

        # ---------- Timeline ----------
        with tab2:
            st.subheader("Timeline Analysis")

            if df_tl.empty:
                st.warning("No data available for selected subreddits.")
            else:
                # Daily sentiment with rolling avg and event markers
                st.markdown("### Daily Sentiment with 7-Day Rolling Average and Event Markers")

                daily = (
                    df_tl.resample("D", on="date")["sentiment_score"]
                    .mean()
                    .reset_index()
                )

                if daily.empty:
                    st.warning("Not enough data for daily sentiment.")
                else:
                    daily["rolling_7d"] = daily["sentiment_score"].rolling(
                        window=7, min_periods=1
                    ).mean()

                    fig_ts = px.line(
                        daily,
                        x="date",
                        y="sentiment_score",
                        title="Daily Sentiment (All Selected Subreddits)",
                    )
                    fig_ts.add_scatter(
                        x=daily["date"],
                        y=daily["rolling_7d"],
                        mode="lines",
                        name="7-day Rolling Avg",
                    )
                    fig_ts.add_hline(y=0, line_dash="dash")
                    fig_ts.update_layout(hovermode="x unified")

                    # Embed sentiment spike events as markers
                    sent_spikes = events.get("sentiment_spikes", pd.DataFrame())
                    if not sent_spikes.empty:
                        fig_ts.add_scatter(
                            x=sent_spikes["date"],
                            y=sent_spikes["sentiment_mean"],
                            mode="markers",
                            name="Sentiment Spike Events",
                            marker=dict(size=11, symbol="star"),
                        )

                    st.plotly_chart(fig_ts, use_container_width=True)

                # Sentiment by subreddit over time
                st.markdown("### Sentiment by Subreddit Over Time")

                by_sub_daily = (
                    df_tl.groupby(
                        [pd.Grouper(key="date", freq="D"), "subreddit"]
                    )["sentiment_score"]
                    .mean()
                    .reset_index()
                )

                if by_sub_daily.empty:
                    st.warning("Not enough data for subreddit comparison timeline.")
                else:
                    fig_comp = px.line(
                        by_sub_daily,
                        x="date",
                        y="sentiment_score",
                        color="subreddit",
                        title="Daily Sentiment by Subreddit",
                    )
                    fig_comp.add_hline(y=0, line_dash="dash")
                    fig_comp.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_comp, use_container_width=True)

                # Emotion distribution over time
                st.markdown("### Emotion Distribution Over Time (All Selected Subreddits)")
                emo_daily = (
                    df_tl.groupby([pd.Grouper(key="date", freq="D"), "emotion"])
                    .size()
                    .reset_index(name="count")
                )

                if emo_daily.empty:
                    st.warning("Not enough data for emotion timeline.")
                else:
                    fig_emo = px.area(
                        emo_daily,
                        x="date",
                        y="count",
                        color="emotion",
                        title="Emotion Distribution Over Time",
                    )
                    fig_emo.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_emo, use_container_width=True)

                # Emotion variance over time
                st.markdown("### Emotion Variance (Entropy) Over Time")
                emo_list = (
                    df_tl.groupby(pd.Grouper(key="date", freq="D"))["emotion"]
                    .apply(list)
                    .reset_index()
                )
                emo_list["entropy"] = emo_list["emotion"].apply(
                    calculate_emotion_entropy
                )

                if emo_list.empty:
                    st.warning("Not enough data for emotion variance.")
                else:
                    fig_ent = px.line(
                        emo_list,
                        x="date",
                        y="entropy",
                        title="Daily Emotion Entropy (All Selected Subreddits)",
                    )
                    st.plotly_chart(fig_ent, use_container_width=True)

        # ---------- Word Cloud ----------
        with tab3:
            st.subheader("Word Cloud")

            if filtered_df.empty:
                st.warning("No data available for selected subreddits.")
            else:
                combined_text = " ".join(filtered_df["full_text"].dropna())
                if not combined_text.strip():
                    st.warning("Not enough text for word cloud.")
                else:
                    wc_fig = generate_wordcloud(
                        filtered_df["full_text"],
                        "Word Cloud (Selected Subreddits)",
                    )
                    if wc_fig is not None:
                        st.pyplot(wc_fig)

        # ---------- Review Needed ----------
        with tab4:
            st.subheader("Posts Needing Review")

            if filtered_df.empty:
                st.warning("No data available for selected subreddits.")
            else:
                review_df = filtered_df[filtered_df["needs_review"] == True]

                if review_df.empty:
                    st.success("No posts flagged as needing review.")
                else:
                    st.dataframe(
                        review_df[["subreddit", "emotion", "sentiment", "text", "url"]],
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download Review Items",
                        review_df.to_csv(index=False).encode("utf-8"),
                        file_name="posts_needing_review.csv",
                        mime="text/csv",
                    )

        # ---------- Event Dashboard ----------
        with tab5:
            st.subheader("Event Dashboard")

            if df_tl.empty:
                st.warning("No data available for selected subreddits.")
            else:
                sent_spikes = events.get("sentiment_spikes", pd.DataFrame())
                act_spikes = events.get("activity_spikes", pd.DataFrame())
                emo_shifts = events.get("emotion_shifts", pd.DataFrame())

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Sentiment Spike Events",
                        len(sent_spikes),
                    )
                with col2:
                    st.metric(
                        "Activity Spike Events",
                        len(act_spikes),
                    )
                with col3:
                    st.metric(
                        "Emotion Shift Events",
                        len(emo_shifts),
                    )

                # Combined log
                all_events = []

                if not sent_spikes.empty:
                    for _, r in sent_spikes.iterrows():
                        all_events.append(
                            {
                                "date": r["date"],
                                "type": "Sentiment Spike",
                                "detail": f"Mean sentiment={r['sentiment_mean']:.3f}, z={r['z']:.2f}",
                            }
                        )

                if not act_spikes.empty:
                    for _, r in act_spikes.iterrows():
                        all_events.append(
                            {
                                "date": r["date"],
                                "type": "Activity Spike",
                                "detail": f"Posts={r['post_count']}, z={r['z']:.2f}",
                            }
                        )

                if not emo_shifts.empty:
                    for _, r in emo_shifts.iterrows():
                        all_events.append(
                            {
                                "date": r["date"],
                                "type": "Emotion Shift",
                                "detail": f"KL divergence={r['kl_divergence']:.3f}",
                            }
                        )

                if not all_events:
                    st.info("No major events detected for selected subreddits.")
                else:
                    events_df = pd.DataFrame(all_events).sort_values("date")
                    st.subheader("Event Log")
                    st.dataframe(events_df, use_container_width=True)


if __name__ == "__main__":
    main()
