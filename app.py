import json
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
    "advanced statistics, and team context — and then lets you fold in real-world context "
    "(injuries, trades, role changes) that the stats-only model can't see."
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
    return pd.read_csv("final_nba_projection_output.csv")


@st.cache_resource
def load_model_and_explainer():
    model = joblib.load("nba_xgboost_model.pkl")
    feature_names = joblib.load("model_features.pkl")
    explainer = shap.TreeExplainer(model)
    return model, feature_names, explainer


@st.cache_data
def compute_empirical_band(_data):
    """80% prediction interval from the empirical residual distribution."""
    errors = _data["ERROR"].dropna()
    return float(errors.quantile(0.10)), float(errors.quantile(0.90))


@st.cache_resource
def build_nn_index(_data, _feature_names):
    """Standardized k-NN index for similar historical player-seasons."""
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

missing_features = [f for f in feature_names if f not in data.columns]
features_complete = len(missing_features) == 0

# ------------------------------------------------------------
# Sidebar input
# ------------------------------------------------------------

st.sidebar.header("User Input")

players = sorted(data["PLAYER_NAME"].dropna().unique())
selected_player = st.sidebar.selectbox("Select a player:", players)
use_claude = st.sidebar.checkbox("Use Claude-generated explanation", value=True)
generate_button = st.sidebar.button("Generate Projection")

if not features_complete:
    st.sidebar.warning(
        f"{len(missing_features)} model features missing from CSV. "
        "SHAP and similar-player comps will be limited until you re-export "
        "`final_nba_projection_output.csv` with the full feature matrix."
    )

# ------------------------------------------------------------
# Anthropic client helper
# ------------------------------------------------------------

def get_anthropic_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


# ------------------------------------------------------------
# Confidence range, context, and SHAP helpers
# ------------------------------------------------------------

def get_confidence_range(predicted):
    """80% empirical prediction interval centered on a given prediction."""
    return predicted + band_lower, predicted + band_upper


def build_context_lines(row):
    snap_cols = [
        ("AGE", "Age"), ("GP", "Games played"), ("MIN", "Minutes played"),
        ("USG_PCT", "Usage rate"), ("TS_PCT", "True shooting %"),
        ("TEAM_PACE", "Team pace"), ("TEAM_OFF_RATING", "Team offensive rating"),
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


# ------------------------------------------------------------
# AI COMPONENT: News-aware projection adjustment
# ------------------------------------------------------------

def extract_news_signal(news_text: str, player_name: str) -> dict:
    """
    Use Claude as a structured-extraction layer over noisy free-text input.
    Returns a dict with quantitative adjustment signals or has_relevant_info=False.

    Handles real-world messiness:
      - Empty / whitespace-only input
      - Excessively long input (truncated)
      - Off-topic input or news about a different player (caught by the prompt)
      - Malformed JSON output (caught and reported)
      - Missing API key (clean fallback)
      - Out-of-range numeric values (clamped downstream)
    """
    if not news_text or not news_text.strip():
        return {"has_relevant_info": False, "reason": "No input provided."}

    news_text = news_text.strip()[:5000]

    client = get_anthropic_client()
    if client is None:
        return {"has_relevant_info": False, "reason": "ANTHROPIC_API_KEY not configured."}

    prompt = f"""You are an NBA analyst extracting structured signals from a free-text news/report blob about a specific player.

Target player: {player_name}

Input text:
\"\"\"
{news_text}
\"\"\"

Extract any FORWARD-LOOKING information about this player that would change a points projection for next season. Examples of relevant info: injuries that will cause missed games, surgeries with recovery timelines, trades, role changes (starter to bench or vice versa), team tanking, contract holdouts, retirement.

Output ONLY a single JSON object with this exact schema (no preamble, no markdown fences, no comments):
{{
  "has_relevant_info": <true|false>,
  "expected_games_missed": <integer 0-82, conservative>,
  "minutes_per_game_change_pct": <number, e.g. -10 for 10% fewer mpg, +5 for 5% more>,
  "usage_role_change_pct": <number, e.g. +15 for a much bigger scoring role>,
  "severity": "<low|medium|high>",
  "extraction_confidence": <0.0 to 1.0>,
  "reasoning": "<one or two sentences>"
}}

Rules:
- If the text has no forward-looking, prediction-relevant info about THIS player, set has_relevant_info=false and all numeric fields to 0.
- If the text is about a different player, set has_relevant_info=false.
- Be conservative on games missed — only count games for which there is direct evidence in the text.
- Do not invent information that is not in the text.
- Output STRICT JSON only.
"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip("` \n")
        signal = json.loads(raw)
        if "has_relevant_info" not in signal:
            return {"has_relevant_info": False, "reason": "Model output missing required fields."}
        return signal
    except json.JSONDecodeError:
        return {"has_relevant_info": False, "reason": "Model returned non-JSON output."}
    except Exception as e:
        return {"has_relevant_info": False, "reason": f"API error: {e}"}


def apply_adjustment(baseline_pts: float, signal: dict) -> dict:
    """
    Convert structured signal into a multiplicative adjustment of the baseline
    XGBoost prediction. Three independent factors:
      availability  = (82 - games_missed) / 82       (range [0, 1])
      minutes       = 1 + minutes_per_game_change_pct/100  (clamped 0.5-1.5)
      role          = 1 + usage_role_change_pct/100        (clamped 0.5-1.5)
    Final adjusted = baseline * availability * minutes * role
    """
    if not signal.get("has_relevant_info"):
        return {"applied": False, "baseline": baseline_pts, "adjusted": baseline_pts}

    games_missed = int(np.clip(signal.get("expected_games_missed", 0) or 0, 0, 82))
    mpg_pct = float(signal.get("minutes_per_game_change_pct", 0) or 0)
    role_pct = float(signal.get("usage_role_change_pct", 0) or 0)

    availability = (82 - games_missed) / 82
    minutes_factor = float(np.clip(1 + mpg_pct / 100, 0.5, 1.5))
    role_factor = float(np.clip(1 + role_pct / 100, 0.5, 1.5))

    adjusted = max(0.0, baseline_pts * availability * minutes_factor * role_factor)

    return {
        "applied": True,
        "baseline": float(baseline_pts),
        "adjusted": float(adjusted),
        "factors": {
            "availability": availability,
            "minutes": minutes_factor,
            "role": role_factor,
        },
        "components": {
            "games_missed": games_missed,
            "mpg_change_pct": mpg_pct,
            "role_change_pct": role_pct,
        },
        "severity": signal.get("severity", "low"),
        "confidence": float(signal.get("extraction_confidence", 0) or 0),
        "reasoning": signal.get("reasoning", ""),
    }


# ------------------------------------------------------------
# Narrative explanations
# ------------------------------------------------------------

def generate_template_explanation(row, predicted_pts=None):
    player_name = row["PLAYER_NAME"]
    current_pts = row["PTS"]
    if predicted_pts is None:
        predicted_pts = row["PREDICTED_NEXT_SEASON_PTS"]
    lower, upper = get_confidence_range(predicted_pts)

    if predicted_pts > current_pts:
        direction = "increase"
    elif predicted_pts < current_pts:
        direction = "decrease"
    else:
        direction = "remain about the same"

    return f"""
{player_name} scored **{current_pts:,.0f} points** in the 2023-24 season. 
The model projects about **{predicted_pts:,.0f} points** the following season, 
with an estimated 80% range of **{lower:,.0f} to {upper:,.0f} points**.

Scoring is expected to **{direction}**. The projection is based on patterns from historical
NBA player seasons (scoring volume, games played, minutes, usage rate, shooting efficiency,
pace, offensive rating, and year-over-year role changes).
"""


def generate_claude_explanation(row):
    player_name = row["PLAYER_NAME"]
    current_pts = row["PTS"]
    predicted_pts = row["PREDICTED_NEXT_SEASON_PTS"]
    actual_pts = row["NEXT_SEASON_PTS"]
    abs_error = row["ABS_ERROR"]
    lower, upper = get_confidence_range(predicted_pts)

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

Top SHAP drivers for this prediction:
{shap_lines or "(SHAP unavailable)"}

In paragraph 1, explain the projection and call out specific drivers above (a notable change in
usage rate, minutes, games played, or a top SHAP feature) that justify whether the projection
is up, down, or stable relative to last season. Be specific about which numbers matter.

In paragraph 2, be honest about uncertainty. Mention that the model uses historical patterns and
that real-world factors like injuries, trades, role changes, and breakout seasons can move the
actual outcome. Do not give betting advice. Do not overstate certainty.
"""

    client = get_anthropic_client()
    if client is None:
        return generate_template_explanation(row)
    try:
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception:
        return generate_template_explanation(row)


# ------------------------------------------------------------
# Render helpers
# ------------------------------------------------------------

def render_shap_explanation(row):
    if not features_complete:
        preview = ", ".join(missing_features[:8])
        more = f" and {len(missing_features) - 8} more" if len(missing_features) > 8 else ""
        st.info(
            f"SHAP plot unavailable: missing model features in CSV ({preview}{more}).\n\n"
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
    if nn_index is None:
        st.info("Similar-player comps unavailable: not enough model features in CSV.")
        return

    X_player = row[nn_features].fillna(0).astype(float).values.reshape(1, -1)
    X_scaled = nn_scaler.transform(X_player)
    _, indices = nn_index.kneighbors(X_scaled)
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
        "Historical player-seasons whose feature profile (age, role, efficiency, team context, "
        "year-over-year trajectory) is closest to the selected player's. Useful as a sanity check."
    )


def render_news_adjustment(row):
    """The headline AI component: structured extraction → ML pipeline adjustment."""
    st.write(
        "The XGBoost model is trained on stats alone — it has no awareness of injuries, trades, "
        "or role changes that haven't happened yet. Paste any recent news, injury report, or "
        "forward-looking context below. Claude will extract a structured signal "
        "(games missed, minutes change, usage change) and apply it as a calibrated adjustment "
        "factor on top of the baseline ML prediction."
    )

    news_text = st.text_area(
        f"Forward-looking context about {row['PLAYER_NAME']} (optional):",
        value="",
        height=130,
        key=f"news_input_{row['PLAYER_NAME']}",
        placeholder=(
            "Example: 'Joel Embiid underwent meniscus surgery in February and the team has "
            "indicated he is expected to miss the first two months of the 2024-25 season as part "
            "of a load-management plan.'"
        ),
    )

    if st.button("Adjust Projection with Context", key="adjust_btn"):
        if not news_text.strip():
            st.warning("Paste some context first, or skip this section.")
            return

        with st.spinner("Claude is extracting structured signals from the text..."):
            signal = extract_news_signal(news_text, row["PLAYER_NAME"])

        if not signal.get("has_relevant_info"):
            reason = signal.get("reason") or signal.get("reasoning") or "No actionable signal."
            st.info(f"No actionable forward-looking signal extracted. _{reason}_")
            return

        baseline = float(row["PREDICTED_NEXT_SEASON_PTS"])
        adj = apply_adjustment(baseline, signal)

        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            st.metric("Baseline (model only)", f"{adj['baseline']:,.0f}")
        with ac2:
            delta = adj["adjusted"] - adj["baseline"]
            st.metric(
                "Adjusted (model + context)",
                f"{adj['adjusted']:,.0f}",
                delta=f"{delta:+,.0f}",
            )
        with ac3:
            ratio = adj["adjusted"] / adj["baseline"] if adj["baseline"] else 1.0
            st.metric("Net adjustment", f"{ratio:.2f}×")

        lo_adj, hi_adj = get_confidence_range(adj["adjusted"])
        st.write(
            f"**Adjusted 80% range:** {lo_adj:,.0f} to {hi_adj:,.0f} points  \n"
            f"Extraction confidence: {adj['confidence']:.0%} | Severity: {adj['severity']}"
        )

        comp = adj["components"]
        facs = adj["factors"]
        breakdown = pd.DataFrame({
            "Adjustment": ["Availability", "Minutes per game", "Usage / role"],
            "Extracted signal": [
                f"{comp['games_missed']} games missed",
                f"{comp['mpg_change_pct']:+.0f}% mpg",
                f"{comp['role_change_pct']:+.0f}% usage",
            ],
            "Multiplier": [
                f"{facs['availability']:.2f}×",
                f"{facs['minutes']:.2f}×",
                f"{facs['role']:.2f}×",
            ],
        })
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

        with st.expander("Claude's reasoning"):
            st.write(adj["reasoning"] or "(no reasoning returned)")

        with st.expander("Raw extracted signal (debug)"):
            st.json(signal)


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

    lower, upper = get_confidence_range(player_row["PREDICTED_NEXT_SEASON_PTS"])
    st.write("### 80% Prediction Range")
    st.write(
        f"The model estimates a likely range of **{lower:,.0f} to {upper:,.0f} points**. "
        "This is an 80% interval derived from the empirical distribution of model errors on the "
        "test set — about 80% of historical predictions fell within a band of this width."
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

    st.write("### 🔍 Real-World Context Adjustment")
    render_news_adjustment(player_row)

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
        f"Predictions were off by about **{MAE:.0f} points on average**. "
        f"R² of **{R2:.3f}** means the model explains about 67.1% of the variation in "
        "next-season point totals on the test set."
    )

    with st.expander("View Raw Prediction Data"):
        st.dataframe(player_row.to_frame().T)

else:
    st.info("Select a player from the sidebar and click **Generate Projection**.")