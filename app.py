import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import shap
from anthropic import Anthropic

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="NBA Player Points Projection Tool",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 NBA Player Points Projection Tool")
st.write(
    "This tool predicts a player's next-season total points using historical NBA performance data, "
    "advanced statistics, and team context."
)

# ------------------------------------------------------------
# Model evaluation metrics from training
# ------------------------------------------------------------

MAE = 218.62
RMSE = 289.11
R2 = 0.671

# ------------------------------------------------------------
# Load data, model, and build helper indexes
# ------------------------------------------------------------

@st.cache_data
def load_data():
    # Use the richer projection output that includes feature columns.
    return pd.read_csv("final_nba_projection_output.csv")


@st.cache_resource
def load_model_and_explainer():
    model = joblib.load("nba_xgboost_model.pkl")
    feature_names = joblib.load("model_features.pkl")
    explainer = shap.TreeExplainer(model)
    return model, feature_names, explainer


@st.cache_data
def compute_empirical_band(_data):
    """
    80% prediction interval derived from the empirical distribution of
    residuals (actual - predicted) on the test set. This replaces the
    previous +/- MAE band, which only covered ~60% of actual values.
    """
    errors = _data["ERROR"].dropna()
    return float(errors.quantile(0.10)), float(errors.quantile(0.90))


@st.cache_resource
def build_nn_index(_data, _feature_names):
    """Standardized k-NN index for finding similar historical player-seasons."""
    available = [f for f in _feature_names if f in _data.columns]
    if len(available) < 5:
        return None, None, []
    X = _data[available].fillna(0).astype(float).values
    scaler = StandardScaler().fit(X)
    nn = NearestNeighbors(n_neighbors=6).fit(scaler.transform(X))
    return nn, scaler, available


data = load_data()
model, feature_names, explainer = load_model_and_explainer()
band_lower, band_upper = compute_empirical_band(data)
nn_index, nn_scaler, nn_features = build_nn_index(data, feature_names)

# Track which model features are present in the CSV for graceful degradation.
missing_features = [f for f in feature_names if f not in data.columns]
features_complete = len(missing_features) == 0

# ------------------------------------------------------------
# Sidebar input
# ------------------------------------------------------------

st.sidebar.header("User Input")

players = sorted(data["PLAYER_NAME"].dropna().unique())

selected_player = st.sidebar.selectbox(
    "Select a player:",
    players
)

use_claude = st.sidebar.checkbox(
    "Use Claude-generated explanation",
    value=True
)

generate_button = st.sidebar.button("Generate Projection")

if not features_complete:
    st.sidebar.warning(
        f"{len(missing_features)} model features are missing from the CSV. "
        "SHAP and similar-player comps run on whichever features are present, "
        "but for full fidelity re-export `final_nba_projection_output.csv` "
        "from your training notebook with the complete feature matrix."
    )

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def get_confidence_range(row):
    """80% empirical prediction interval (asymmetric, calibrated to past errors)."""
    predicted = row["PREDICTED_NEXT_SEASON_PTS"]
    return predicted + band_lower, predicted + band_upper


def build_context_lines(row):
    """Pull contextual stats and YoY changes from the row, when available."""
    snap_cols = [
        ("AGE", "Age"),
        ("GP", "Games played"),
        ("MIN", "Minutes played"),
        ("USG_PCT", "Usage rate"),
        ("TS_PCT", "True shooting %"),
        ("TEAM_PACE", "Team pace"),
        ("TEAM_OFF_RATING", "Team offensive rating"),
    ]
    delta_cols = [
        ("PTS_CHANGE", "Year-over-year points change"),
        ("MIN_CHANGE", "Year-over-year minutes change"),
        ("GP_CHANGE", "Year-over-year games-played change"),
        ("USG_CHANGE", "Year-over-year usage rate change"),
        ("TS_CHANGE", "Year-over-year true shooting change"),
    ]
    lines = []
    for col, label in snap_cols:
        if col in row.index and pd.notna(row[col]):
            lines.append(f"- {label}: {row[col]:.2f}")
    for col, label in delta_cols:
        if col in row.index and pd.notna(row[col]):
            lines.append(f"- {label}: {row[col]:+.2f}")
    return "\n".join(lines)


def top_shap_drivers(row, k=3):
    """Return the top-k absolute SHAP contributions as text bullets for the Claude prompt."""
    if not features_complete:
        return ""
    X_player = pd.DataFrame([row[feature_names].astype(float).values], columns=feature_names)
    shap_values = explainer(X_player)
    contribs = shap_values.values[0]
    order = np.argsort(np.abs(contribs))[::-1][:k]
    bullets = []
    for i in order:
        feat = feature_names[i]
        val = float(row[feat]) if feat in row.index else float("nan")
        bullets.append(f"- {feat} = {val:.2f} (SHAP impact {contribs[i]:+.1f} pts)")
    return "\n".join(bullets)


def generate_template_explanation(row):
    player_name = row["PLAYER_NAME"]
    current_pts = row["PTS"]
    predicted_pts = row["PREDICTED_NEXT_SEASON_PTS"]
    lower, upper = get_confidence_range(row)

    if predicted_pts > current_pts:
        direction = "increase"
        interpretation = "The model expects the player's scoring output to rise compared with the previous season."
    elif predicted_pts < current_pts:
        direction = "decrease"
        interpretation = "The model expects the player's scoring output to decline compared with the previous season."
    else:
        direction = "remain about the same"
        interpretation = "The model expects the player's scoring output to remain relatively stable."

    summary = f"""
{player_name} scored **{current_pts:,.0f} points** in the 2023-24 season. 
The model projects that he will score about **{predicted_pts:,.0f} points** the following season, 
with an estimated 80% range of **{lower:,.0f} to {upper:,.0f} points**.

This suggests that his scoring is expected to **{direction}**. {interpretation}

The projection is based on patterns from historical NBA player seasons, including prior scoring volume, 
games played, minutes, usage rate, shooting efficiency, pace, offensive rating, and year-over-year role changes. 
Factors such as injuries, trades, coaching changes, and breakout seasons may cause real results to differ from the model.
"""
    return summary


def generate_claude_explanation(row):
    player_name = row["PLAYER_NAME"]
    current_pts = row["PTS"]
    predicted_pts = row["PREDICTED_NEXT_SEASON_PTS"]
    actual_pts = row["NEXT_SEASON_PTS"]
    abs_error = row["ABS_ERROR"]
    lower, upper = get_confidence_range(row)

    context_lines = build_context_lines(row)
    shap_lines = top_shap_drivers(row, k=3)

    prompt = f"""
You are explaining an NBA player points projection model to a non-technical user.

Write a concise, professional explanation in 2 short paragraphs.

Player: {player_name}
Previous season points: {current_pts:.0f}
Predicted next-season points: {predicted_pts:.0f}
Estimated 80% range: {lower:.0f} to {upper:.0f}
Actual next-season points (historical evaluation): {actual_pts:.0f}
Absolute model error on this case: {abs_error:.0f}
Model MAE: {MAE:.2f}, RMSE: {RMSE:.2f}, R-squared: {R2:.3f}

Player context:
{context_lines or "(no extra context available)"}

Top SHAP drivers for this prediction (these are the features that moved the model most):
{shap_lines or "(SHAP unavailable)"}

In paragraph 1, explain the projection and call out the most informative drivers above
(for example, a notable change in usage rate, minutes, or games played, or a top SHAP feature)
that justify whether the projection is up, down, or stable relative to last season.
Be specific about which numbers matter.

In paragraph 2, be honest about uncertainty. Mention that the model uses historical player-season
patterns (points, minutes, games, usage, true shooting, pace, offensive rating, year-over-year
changes) and that real-world factors like injuries, trades, role changes, coaching decisions, and
breakout seasons can move the actual outcome. Do not give betting advice. Do not overstate certainty.
"""

    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if not api_key:
            return generate_template_explanation(row)

        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    except Exception:
        return generate_template_explanation(row)


def render_shap_explanation(row):
    """Per-prediction SHAP waterfall plot."""
    if not features_complete:
        preview = ", ".join(missing_features[:8])
        more = f" and {len(missing_features) - 8} more" if len(missing_features) > 8 else ""
        st.info(
            "SHAP plot unavailable: the CSV is missing model features required for "
            f"explanation: {preview}{more}.\n\n"
            "To enable: re-export `final_nba_projection_output.csv` from your training "
            "notebook with the full feature matrix included."
        )
        return

    X_player = pd.DataFrame([row[feature_names].astype(float).values], columns=feature_names)
    shap_values = explainer(X_player)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Each bar shows how much that feature pushed the prediction up (red) or down (blue) "
        "from the model's average prediction. The final value is the projected total points."
    )


def render_similar_players(row, current_idx):
    """Top-5 historical player-seasons with the most similar feature profile."""
    if nn_index is None:
        st.info("Similar-player comps unavailable: not enough model features in CSV.")
        return

    X_player = row[nn_features].fillna(0).astype(float).values.reshape(1, -1)
    X_scaled = nn_scaler.transform(X_player)
    _, indices = nn_index.kneighbors(X_scaled)

    # Drop the player themselves if they appear in the neighbor set.
    similar_idx = [int(i) for i in indices[0] if int(i) != int(current_idx)][:5]
    if not similar_idx:
        st.info("No comps found.")
        return

    cols = ["PLAYER_NAME", "SEASON", "TEAM_ABBREVIATION", "PTS",
            "NEXT_SEASON_PTS", "PREDICTED_NEXT_SEASON_PTS", "ABS_ERROR"]
    cols = [c for c in cols if c in data.columns]
    comps = data.iloc[similar_idx][cols].rename(columns={
        "PTS": "Prev Season Pts",
        "NEXT_SEASON_PTS": "Actual Next-Season Pts",
        "PREDICTED_NEXT_SEASON_PTS": "Model Prediction",
        "ABS_ERROR": "Model Abs Error",
    })
    st.dataframe(comps, use_container_width=True, hide_index=True)
    st.caption(
        "These are the historical player-seasons whose feature profile (age, role, efficiency, "
        "team context, year-over-year trajectory) is closest to the selected player's. "
        "Useful as a sanity check on the projection."
    )


# ------------------------------------------------------------
# Main output
# ------------------------------------------------------------

if generate_button:
    matches = data.index[data["PLAYER_NAME"] == selected_player]
    player_idx = int(matches[0])
    player_row = data.iloc[player_idx]

    st.subheader(f"Projection for {selected_player}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Previous Season Points", f"{player_row['PTS']:,.0f}")
    with col2:
        st.metric("Predicted Next-Season Points", f"{player_row['PREDICTED_NEXT_SEASON_PTS']:,.0f}")
    with col3:
        st.metric("Actual Points (Evaluation)", f"{player_row['NEXT_SEASON_PTS']:,.0f}")

    st.divider()

    lower, upper = get_confidence_range(player_row)
    st.write("### 80% Prediction Range")
    st.write(
        f"The model estimates a likely range of **{lower:,.0f} to {upper:,.0f} points**. "
        "This is an 80% interval derived from the empirical distribution of model errors "
        "on the test set (calibrated, not assumed). About 80% of historical predictions fell "
        "within a band of this width."
    )

    st.write("### Model Error on This Case")
    st.write(
        f"For this historical test case, the model was off by about "
        f"**{player_row['ABS_ERROR']:,.0f} points**."
    )

    st.divider()

    st.write("### Claude-Generated Projection Explanation")
    explanation = (
        generate_claude_explanation(player_row) if use_claude
        else generate_template_explanation(player_row)
    )
    st.markdown(explanation)

    st.divider()

    st.write("### Why the Model Made This Projection")
    render_shap_explanation(player_row)

    st.divider()

    st.write("### Most Similar Historical Player-Seasons")
    render_similar_players(player_row, player_idx)

    st.divider()

    st.write("### Model Evaluation")
    st.write(
        "The model was trained on historical NBA player-season data and tested by using "
        "2023-24 player data to predict 2024-25 total points."
    )
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Mean Absolute Error", f"{MAE:.2f}")
    with m2:
        st.metric("RMSE", f"{RMSE:.2f}")
    with m3:
        st.metric("R²", f"{R2:.3f}")

    st.write(
        f"The mean absolute error means predictions were off by about **{MAE:.0f} points on average**. "
        f"The R² of **{R2:.3f}** means the model explains about 67.1% of the variation in "
        "next-season point totals on the test set."
    )

    with st.expander("View Raw Prediction Data"):
        st.dataframe(player_row.to_frame().T)

else:
    st.info("Select a player from the sidebar and click **Generate Projection**.")