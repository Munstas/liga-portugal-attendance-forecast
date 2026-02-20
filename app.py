# app.py - ENHANCED VERSION
import streamlit as st
import pandas as pd
import numpy as np

from final_model import load_final_model, predict_attendance
from predict import get_team_stats, get_matchup_stats, is_big3

# Page config
st.set_page_config(
    page_title="Liga Portugal Attendance Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and model
@st.cache_data
def load_features_data():
    return pd.read_csv("data/liga_portugal_features.csv")

@st.cache_resource
def load_trained_model():
    return load_final_model()

df = load_features_data()
model, label_encoder, feature_cols = load_trained_model()

# Sidebar
st.sidebar.title("⚙️ Filters")
teams = sorted(set(df["Home"]) | set(df["Away"]))
home_team = st.sidebar.selectbox("🏠 Home Team", teams)
away_team = st.sidebar.selectbox("✈️ Away Team", [t for t in teams if t != home_team])
round_num = st.sidebar.selectbox("🔢 Round", sorted(df["Round_Num"].unique()))
day_type = st.sidebar.selectbox("📅 Day Type", sorted(df["Day_Type"].unique()))

if st.sidebar.button("🎯 Predict Attendance", use_container_width=True, type="primary"):
    st.session_state.prediction_made = True
else:
    st.session_state.prediction_made = False

# Main page
st.title("⚽ Liga Portugal - Attendance Predictor")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"### {home_team} vs {away_team}")
with col2:
    st.metric("Overall Average", f"{df['Attendance'].mean():,.0f}")

# History
st.subheader("📊 Historical Data")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Head-to-Head**")
    mask = ((df["Home"] == home_team) & (df["Away"] == away_team))
    df_hist = df[mask].sort_values("Date")
    if not df_hist.empty:
        st.metric("Matchup Average", f"{int(df_hist['Attendance'].mean()):,}")
        st.line_chart(df_hist.set_index("Date")["Attendance"])
    else:
        st.info("No previous matches between these teams")

with col2:
    st.markdown("**Home Recent Games**")
    df_recent = df[df["Home"] == home_team].tail(6).sort_values("Date")
    if not df_recent.empty:
        st.metric("Recent Home Average", f"{int(df_recent['Attendance'].mean()):,}")
        st.line_chart(df_recent.set_index("Date")["Attendance"])

# Prediction
st.subheader("🎯 Prediction")
if st.session_state.get("prediction_made", False):
    home_stats = get_team_stats(home_team)
    matchup_avg = get_matchup_stats(home_team, away_team)
    
    prediction = predict_attendance(
        home_team, away_team, round_num, day_type,
        home_stats['home_avg'], home_stats['home_last3'],
        matchup_avg, is_big3(home_team), is_big3(away_team)
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Predicted Attendance", f"{prediction:,}")
    with col2:
        st.metric("🏠 Home Team Average", f"{home_stats['home_avg']:,}")
    with col3:
        st.metric("📈 Recent Form", f"{home_stats['home_last3']:,}")
    
    # Confidence interval
    margin = 4000 if is_big3(home_team) else 1000
    low, high = prediction - margin, prediction + margin
    st.success(f"**Confidence Range:** {low:,} - {high:,} (±{margin:,})")
    
    if is_big3(home_team):
        st.info("ℹ️ Big3 teams have higher variance due to match importance and opponent strength")

else:
    st.info("Click 'Predict Attendance' in the sidebar to see the prediction")

st.caption("Developed by André Apolónia | [GitHub](https://github.com/Munstas)")