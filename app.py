import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
# Load data
# ------------------------------------------------------------

@st.cache_data
def load_data():
    results = pd.read_csv("nba_points_predictions_results.csv")
    final_output = pd.read_csv("final_nba_projection_output.csv")
    return results, final_output

results, final_output = load_data()

# ------------------------------------------------------------
# Model evaluation metrics from Colab
# ------------------------------------------------------------

MAE = 218.62
RMSE = 289.11
R2 = 0.671

# ------------------------------------------------------------
# Sidebar input
# ------------------------------------------------------------

st.sidebar.header("User Input")

players = sorted(results["PLAYER_NAME"].dropna().unique())

selected_player = st.sidebar.selectbox(
    "Select a player:",
    players
)

use_claude = st.sidebar.checkbox(
    "Use Claude-generated explanation",
    value=True
)

generate_button = st.sidebar.button("Generate Projection")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def get_confidence_range(row):
    predicted_pts = row["PREDICTED_NEXT_SEASON_PTS"]

    if "LOWER_CONFIDENCE_RANGE" in row.index and "UPPER_CONFIDENCE_RANGE" in row.index:
        lower = row["LOWER_CONFIDENCE_RANGE"]
        upper = row["UPPER_CONFIDENCE_RANGE"]
    else:
        lower = predicted_pts - MAE
        upper = predicted_pts + MAE

    return lower, upper


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
with an estimated confidence range of **{lower:,.0f} to {upper:,.0f} points**.

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

    prompt = f"""
You are explaining an NBA player points projection model to a non-technical user.

Write a concise, professional explanation in 2 short paragraphs.

Player: {player_name}
Previous season points: {current_pts:.0f}
Predicted next-season points: {predicted_pts:.0f}
Estimated confidence range: {lower:.0f} to {upper:.0f}
Actual next-season points for historical evaluation: {actual_pts:.0f}
Absolute model error: {abs_error:.0f}
Model MAE: {MAE:.2f}
Model RMSE: {RMSE:.2f}
Model R-squared: {R2:.3f}

Mention that the model uses historical player-season patterns such as points, minutes, games played, usage rate, true shooting percentage, pace, offensive rating, and year-over-year changes. Also mention that real-world factors such as injuries, trades, role changes, coaching decisions, and breakout seasons can cause uncertainty.
Do not overstate certainty. Do not give betting advice.
"""

    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)

        if not api_key:
            return generate_template_explanation(row)

        client = Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            temperature=0.4,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return message.content[0].text

    except Exception:
        return generate_template_explanation(row)


# ------------------------------------------------------------
# Main output
# ------------------------------------------------------------

if generate_button:
    player_row = results[results["PLAYER_NAME"] == selected_player].iloc[0]

    st.subheader(f"Projection for {selected_player}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Previous Season Points",
            value=f"{player_row['PTS']:,.0f}"
        )

    with col2:
        st.metric(
            label="Predicted Next-Season Points",
            value=f"{player_row['PREDICTED_NEXT_SEASON_PTS']:,.0f}"
        )

    with col3:
        st.metric(
            label="Actual Points for Evaluation",
            value=f"{player_row['NEXT_SEASON_PTS']:,.0f}"
        )

    st.divider()

    lower, upper = get_confidence_range(player_row)

    st.write("### Confidence Range")
    st.write(
        f"The model estimates a likely range of **{lower:,.0f} to {upper:,.0f} points**. "
        f"This range is based on the model's mean absolute error of **{MAE:.2f} points**."
    )

    st.write("### Model Error")
    st.write(
        f"For this historical test case, the model was off by about "
        f"**{player_row['ABS_ERROR']:,.0f} points**."
    )

    st.divider()

    st.write("### Claude-Generated Projection Explanation")

    if use_claude:
        explanation = generate_claude_explanation(player_row)
    else:
        explanation = generate_template_explanation(player_row)

    st.markdown(explanation)

    st.divider()

    st.write("### Model Evaluation")
    st.write(
        "The model was trained on historical NBA player-season data and tested by using "
        "2023-24 player data to predict 2024-25 total points."
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            label="Mean Absolute Error",
            value=f"{MAE:.2f}"
        )

    with metric_col2:
        st.metric(
            label="RMSE",
            value=f"{RMSE:.2f}"
        )

    with metric_col3:
        st.metric(
            label="R²",
            value=f"{R2:.3f}"
        )

    st.write(
        "The mean absolute error means the model's predictions were off by about "
        f"**{MAE:.0f} points on average**. The R² value of **{R2:.3f}** means the model explains "
        "about 67.1% of the variation in next-season point totals in the test set."
    )

    with st.expander("View Raw Prediction Data"):
        st.dataframe(player_row.to_frame().T)

else:
    st.info("Select a player from the sidebar and click **Generate Projection**.")