import time
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import praw
import streamlit as st


# -----------------------------------------
# UNIVERSAL TIMESTAMP PARSER
# -----------------------------------------
def parse_timestamp(x):
    """
    Convert various timestamp formats into pandas.Timestamp.
    Handles:
    - datetime objects
    - Unix seconds
    - Unix milliseconds
    - ISO8601 strings
    Returns NaT for invalid values.
    """
    try:
        # Already datetime
        if isinstance(x, datetime):
            return x

        # Milliseconds since epoch (very large integers)
        if isinstance(x, (int, float)) and x > 10_000_000_000:
            return pd.to_datetime(x, unit="ms", errors="coerce")

        # Seconds since epoch
        if isinstance(x, (int, float)):
            return pd.to_datetime(x, unit="s", errors="coerce")

        # String-like (ISO etc.)
        return pd.to_datetime(str(x), errors="coerce")

    except Exception:
        return pd.NaT


# -----------------------------------------
# REDDIT FETCHER CLASS
# -----------------------------------------
class RedditFetcher:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            ratelimit_seconds=300,
        )
        self.last_request_time = 0.0
        self.request_delay = 2.0  # seconds between API calls

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def set_request_delay(self, delay: float):
        self.request_delay = max(0.0, float(delay))

    def fetch_posts(
        self,
        subreddit_name: str,
        limit: int = 100,
        time_filter: str = "week",
        post_type: str = "top",
    ) -> List[Dict]:
        """
        Fetch posts from a subreddit and return them as a list of dictionaries.
        The created_utc field is stored exactly as provided by Reddit
        (often seconds since epoch), and we normalize later.
        """
        self._rate_limit()

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts: List[Dict] = []

            if post_type == "hot":
                listing = subreddit.hot(limit=limit)
            elif post_type == "new":
                listing = subreddit.new(limit=limit)
            elif post_type == "rising":
                listing = subreddit.rising(limit=limit)
            else:  # "top"
                listing = subreddit.top(limit=limit, time_filter=time_filter)

            for post in listing:
                try:
                    self._rate_limit()

                    posts.append(
                        {
                            "id": post.id,
                            "title": post.title,
                            "text": post.selftext,
                            "author": str(post.author),
                            # can be seconds, use parser later
                            "created_utc": getattr(post, "created_utc", None),
                            "score": post.score,
                            "upvote_ratio": getattr(post, "upvote_ratio", 0.0),
                            "num_comments": post.num_comments,
                            "url": f"https://reddit.com{post.permalink}",
                            "subreddit": str(post.subreddit),
                            "flair": getattr(post, "link_flair_text", ""),
                            "is_original_content": getattr(post, "is_original_content", False),
                            "is_self": post.is_self,
                        }
                    )

                except Exception as e:
                    st.warning(f"Error processing post {getattr(post, 'id', 'unknown')}: {e}")
                    continue

            return posts

        except Exception as e:
            st.error(f"Error fetching posts from r/{subreddit_name}: {e}")
            return []


# -----------------------------------------
# STREAMLIT FETCH UI
# -----------------------------------------
def reddit_fetch_ui() -> Optional[pd.DataFrame]:
    """
    Streamlit UI for fetching Reddit data.
    Returns a DataFrame with a normalized created_utc datetime column (if available).
    """

    if "fetcher_data" not in st.session_state:
        st.session_state.fetcher_data = None

    if "show_api_settings" not in st.session_state:
        st.session_state.show_api_settings = True

    if st.sidebar.button("Toggle API Settings"):
        st.session_state.show_api_settings = not st.session_state.show_api_settings

    # Sidebar: API credentials
    if st.session_state.show_api_settings:
        with st.sidebar.expander("Reddit API Configuration", expanded=True):
            client_id = st.text_input(
                "Client ID",
                value=st.secrets.get("REDDIT_CLIENT_ID", ""),
                type="password",
                key="reddit_client_id",
            )
            client_secret = st.text_input(
                "Client Secret",
                value=st.secrets.get("REDDIT_CLIENT_SECRET", ""),
                type="password",
                key="reddit_client_secret",
            )
            user_agent = st.text_input(
                "User Agent",
                value=st.secrets.get("REDDIT_USER_AGENT", "RedditSense/1.0"),
                key="reddit_user_agent",
            )
    else:
        client_id = st.session_state.get("reddit_client_id", st.secrets.get("REDDIT_CLIENT_ID", ""))
        client_secret = st.session_state.get(
            "reddit_client_secret", st.secrets.get("REDDIT_CLIENT_SECRET", "")
        )
        user_agent = st.session_state.get(
            "reddit_user_agent", st.secrets.get("REDDIT_USER_AGENT", "RedditSense/1.0")
        )

    st.subheader("Fetch Reddit Data")

    # Subreddit list
    subreddits_input = st.text_area(
        "Subreddits (one per line, max 10)",
        "technology\napple\nworldnews",
        height=100,
        help="Enter one subreddit name per line (no 'r/').",
    )
    subreddits = [s.strip() for s in subreddits_input.splitlines() if s.strip()][:10]

    col1, col2 = st.columns(2)
    with col1:
        post_type = st.selectbox(
            "Post Type",
            ["top", "hot", "new", "rising"],
            index=0,
        )
    with col2:
        time_filter = st.selectbox(
            "Time Period (for 'top')",
            ["all", "year", "month", "week", "day"],
            index=3,
        )

    with st.expander("Advanced Options"):
        limit = st.slider("Posts per Subreddit", 5, 100, 20, 5)
        delay = st.slider("Delay between requests (seconds)", 1, 5, 2, 1)

    # Fetch button
    if st.button("Fetch Posts", type="primary", use_container_width=True):
        if not all([client_id, client_secret, user_agent]):
            st.error("Please provide all Reddit API credentials.")
            return st.session_state.fetcher_data

        if not subreddits:
            st.error("Please enter at least one subreddit.")
            return st.session_state.fetcher_data

        fetcher = RedditFetcher(client_id, client_secret, user_agent)
        fetcher.set_request_delay(delay)

        all_posts: List[Dict] = []
        progress_text = st.empty()
        progress_bar = st.progress(0.0)

        for i, sub in enumerate(subreddits):
            progress_text.text(f"Fetching posts from r/{sub}...")
            posts = fetcher.fetch_posts(
                subreddit_name=sub,
                limit=limit,
                time_filter=time_filter,
                post_type=post_type,
            )
            all_posts.extend(posts)
            progress_bar.progress((i + 1) / len(subreddits))

        if not all_posts:
            st.warning("No posts were fetched. Please check your input and try again.")
            return st.session_state.fetcher_data

        df = pd.DataFrame(all_posts)

        # Normalize timestamps with universal parser
        if "created_utc" in df.columns:
            df["created_utc"] = df["created_utc"].apply(parse_timestamp)

        st.session_state.fetcher_data = df
        st.success(f"Successfully fetched {len(df)} posts from {len(subreddits)} subreddit(s).")

        return df

    return st.session_state.fetcher_data


if __name__ == "__main__":
    df_test = reddit_fetch_ui()
    if df_test is not None:
        st.write(df_test.head())
