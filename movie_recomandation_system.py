import random
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_FILE = Path(__file__).resolve().parent / "final10.xls"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-top: #10131d;
            --bg-bottom: #1a1631;
            --panel: rgba(255, 255, 255, 0.06);
            --panel-strong: rgba(255, 255, 255, 0.11);
            --line: rgba(255, 255, 255, 0.10);
            --text: #f6f3ff;
            --muted: #c7bfd9;
            --gold: #f7c76b;
            --coral: #ff8f6b;
            --blue: #7aa8ff;
            --mint: #65e4be;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 143, 107, 0.18), transparent 25%),
                radial-gradient(circle at top right, rgba(122, 168, 255, 0.18), transparent 22%),
                linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 30px;
            padding: 2.3rem 2.4rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03)),
                linear-gradient(135deg, rgba(18, 18, 26, 0.92), rgba(52, 29, 70, 0.90));
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 26px 80px rgba(0,0,0,0.28);
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle at 84% 18%, rgba(247, 199, 107, 0.22), transparent 18%),
                radial-gradient(circle at 12% 78%, rgba(122, 168, 255, 0.15), transparent 18%);
            pointer-events: none;
        }

        .hero h1 {
            color: var(--text);
            margin: 0 0 0.8rem 0;
            font-size: 3rem;
            letter-spacing: -0.04em;
            line-height: 1.02;
        }

        .hero p {
            color: rgba(246, 243, 255, 0.82);
            max-width: 780px;
            line-height: 1.65;
            font-size: 1rem;
        }

        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.2rem;
        }

        .hero-tag {
            padding: 0.5rem 0.85rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.08);
            color: var(--text);
            font-size: 0.88rem;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 18px 48px rgba(0,0,0,0.16);
        }

        .metric-card {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            min-height: 132px;
        }

        .metric-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-size: 0.78rem;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            color: var(--text);
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.04em;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: 0.55rem;
        }

        .card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            min-height: 240px;
        }

        .card-title {
            color: var(--text);
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.35rem;
        }

        .card-meta {
            color: var(--gold);
            font-size: 0.86rem;
            margin-bottom: 0.7rem;
            letter-spacing: 0.02em;
        }

        .card-text {
            color: var(--muted);
            line-height: 1.6;
            font-size: 0.94rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.8rem;
            margin-bottom: 0.9rem;
        }

        .chip {
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            background: rgba(101, 228, 190, 0.12);
            border: 1px solid rgba(101, 228, 190, 0.20);
            color: #c5fff0;
            font-size: 0.8rem;
        }

        .section-title {
            color: var(--text);
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
        }

        div[data-testid="stSidebar"] {
            background: rgba(14, 16, 23, 0.92);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df.columns = [column.strip() for column in df.columns]
    genre_columns = [column for column in df.columns if column != "title"]
    for column in genre_columns:
        df[column] = df[column].astype(int)

    df["clean_title"] = df["title"].apply(clean_title)
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").fillna("").iloc[:, 0]
    df["genre_list"] = df.apply(
        lambda row: [genre for genre in genre_columns if row[genre] == 1],
        axis=1,
    )
    df["genre_badge"] = df["genre_list"].apply(lambda genres: " • ".join(genres[:4]) if genres else "No genres tagged")
    df["genre_count"] = df["genre_list"].apply(len)
    return df


def clean_title(raw_title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)", "", str(raw_title)).strip()


def build_user_vector(genre_columns: List[str], selected_genres: List[str]) -> np.ndarray:
    return np.array([1 if genre in selected_genres else 0 for genre in genre_columns], dtype=float).reshape(1, -1)


def compute_recommendations(
    df: pd.DataFrame,
    selected_genres: List[str],
    anchor_title: Optional[str],
    count: int,
) -> pd.DataFrame:
    genre_columns = [column for column in df.columns if column not in {"title", "clean_title", "year", "genre_list", "genre_badge", "genre_count"}]
    feature_matrix = df[genre_columns].values.astype(float)

    if anchor_title:
        anchor_row = df[df["title"] == anchor_title].iloc[0]
        base_vector = anchor_row[genre_columns].values.astype(float)
        if selected_genres:
            user_vector = np.maximum(base_vector, build_user_vector(genre_columns, selected_genres).flatten())
        else:
            user_vector = base_vector
    else:
        user_vector = build_user_vector(genre_columns, selected_genres).flatten()

    if np.sum(user_vector) == 0:
        # fallback to popular broad-spectrum titles when no preference is set
        scored = df.copy()
        scored["match_score"] = scored["genre_count"] / max(scored["genre_count"].max(), 1)
    else:
        similarity = cosine_similarity(feature_matrix, user_vector.reshape(1, -1)).flatten()
        scored = df.copy()
        scored["match_score"] = similarity

    if selected_genres:
        scored["selected_overlap"] = scored["genre_list"].apply(
            lambda genres: len(set(genres).intersection(selected_genres))
        )
    else:
        scored["selected_overlap"] = 0

    if anchor_title:
        scored = scored[scored["title"] != anchor_title]

    scored = scored.sort_values(
        by=["match_score", "selected_overlap", "genre_count", "title"],
        ascending=[False, False, False, True],
    )
    return scored.head(count).copy()


def build_reason(row: pd.Series, selected_genres: List[str], anchor_title: Optional[str]) -> str:
    overlaps = [genre for genre in row["genre_list"] if genre in selected_genres]
    if anchor_title and overlaps:
        return f"Strong match for your selected vibe and similar genre profile to {clean_title(anchor_title)}."
    if anchor_title:
        return f"Shares a close genre signature with {clean_title(anchor_title)}."
    if overlaps:
        return f"Matches {len(overlaps)} of your selected genres: {', '.join(overlaps[:3])}."
    return "Broad catalog pick chosen from the strongest overall genre profile."


def render_hero(total_movies: int, genre_count: int) -> None:
    st.markdown(
        f"""
        <section class="hero">
            <h1>CineMatch AI</h1>
            <p>
                A more cinematic movie recommendation experience built for fast discovery,
                better visual polish, and dependable deployment. Pick a few genres, optionally
                anchor on a movie you already love, and get a curated recommendation grid from a
                catalog of {total_movies:,} films.
            </p>
            <div class="hero-tags">
                <span class="hero-tag">Offline-first recommendations</span>
                <span class="hero-tag">Anchor-title mode</span>
                <span class="hero-tag">Genre intelligence</span>
                <span class="hero-tag">{genre_count} supported genres</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(df: pd.DataFrame, selected_genres: List[str], anchor_title: Optional[str], recommendations: pd.DataFrame) -> None:
    top_match = f"{round(float(recommendations.iloc[0]['match_score']) * 100)}%" if not recommendations.empty else "0%"
    cols = st.columns(4)
    cards = [
        ("Catalog size", f"{len(df):,}", "Movies available in the local recommendation dataset."),
        ("Genres selected", str(len(selected_genres)), "Blend multiple moods to shape the recommendation profile."),
        ("Anchor title", clean_title(anchor_title) if anchor_title else "None", "Use a favorite movie to steer recommendations toward a familiar taste cluster."),
        ("Top match score", top_match, "Cosine similarity between your preference vector and the suggested movie profile."),
    ]
    for col, (label, value, note) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_recommendation_cards(recommendations: pd.DataFrame, selected_genres: List[str], anchor_title: Optional[str]) -> None:
    if recommendations.empty:
        st.info("No matching movies were found for the current filters.")
        return

    columns = st.columns(2, gap="large")
    for index, (_, row) in enumerate(recommendations.iterrows()):
        with columns[index % 2]:
            chips = "".join([f'<span class="chip">{genre}</span>' for genre in row["genre_list"][:5]])
            year_text = f" • {row['year']}" if row["year"] else ""
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">{row['clean_title']}</div>
                    <div class="card-meta">
                        Match score: {round(float(row['match_score']) * 100)}%{year_text}
                    </div>
                    <div class="chip-row">{chips}</div>
                    <div class="card-text">
                        {build_reason(row, selected_genres, anchor_title)}
                        <br/><br/>
                        Genre mix: {row['genre_badge']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_spotlight(df: pd.DataFrame) -> None:
    spotlight = df.sample(1, random_state=random.randint(1, 999999)).iloc[0]
    st.markdown(
        f"""
        <div class="panel">
            <div class="section-title">Tonight's Spotlight</div>
            <div class="card-title">{spotlight['clean_title']}</div>
            <div class="card-meta">{spotlight['year'] if spotlight['year'] else 'Year unavailable'}</div>
            <div class="chip-row">
                {"".join([f'<span class="chip">{genre}</span>' for genre in spotlight['genre_list'][:5]])}
            </div>
            <div class="small-muted">
                A quick discovery pick from the catalog for users who want a fast decision instead of a long search.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    df = load_data()
    genre_columns = [column for column in df.columns if column not in {"title", "clean_title", "year", "genre_list", "genre_badge", "genre_count"}]

    render_hero(len(df), len(genre_columns))

    st.sidebar.markdown("## Discovery Controls")
    st.sidebar.markdown(
        """
        <div class="small-muted">
        Build recommendations from genre taste, a favorite title, or both. The engine stays fully local,
        so it is fast and deployment-friendly.
        </div>
        """,
        unsafe_allow_html=True,
    )

    anchor_title = st.sidebar.selectbox(
        "Start from a favorite movie",
        options=[""] + df["title"].tolist()[:5000],
        help="Optional: choose a known film to guide the recommendation cluster.",
    )
    anchor_title = anchor_title or None

    selected_genres = st.sidebar.multiselect(
        "Pick your preferred genres",
        options=genre_columns,
        default=["Action", "Thriller"] if "Action" in genre_columns and "Thriller" in genre_columns else genre_columns[:2],
    )
    recommendation_count = st.sidebar.slider("How many recommendations?", 4, 12, 8, 1)
    min_genres = st.sidebar.slider("Minimum genres in a movie profile", 1, 6, 1, 1)

    filtered_df = df[df["genre_count"] >= min_genres].copy()
    recommendations = compute_recommendations(filtered_df, selected_genres, anchor_title, recommendation_count)

    st.markdown("")
    left, right = st.columns([1.35, 0.85], gap="large")

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Recommendation Studio</div>
                <div class="small-muted">
                    This upgraded interface is designed to feel more like a discovery product and less like a classroom demo.
                    It focuses on strong defaults, quick interaction, and cleaner recommendation cards.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        render_spotlight(filtered_df)

    st.markdown("")
    render_metrics(filtered_df, selected_genres, anchor_title, recommendations)

    st.markdown("")
    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Your Matches</div>
            <div class="small-muted">
                The ranking blends direct genre overlap and cosine similarity against your active preference vector.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_recommendation_cards(recommendations, selected_genres, anchor_title)


if __name__ == "__main__":
    main()
