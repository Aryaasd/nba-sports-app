import streamlit as st
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import altair as alt

st.set_page_config(page_title="NBA Sports App", layout="wide")

st.title("🏀 NBA Sports App")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Home", "Compare Players"])

# Season selector
season = st.sidebar.selectbox("Season", ["2023-24", "2022-23", "2021-22"], index=0)

# Fetch all NBA players
all_players = players.get_players()
player_names = sorted([p["full_name"] for p in all_players])

def get_player_id(name):
    """Return NBA player ID if found, else None."""
    for p in all_players:
        if p["full_name"].lower() == name.lower():
            return p["id"]
    return None

def get_player_stats(player_name, season):
    """Fetch player game logs for a given season."""
    pid = get_player_id(player_name)
    if pid is None:
        return None
    gamelog = playergamelog.PlayerGameLog(player_id=pid, season=season)
    df = gamelog.get_data_frames()[0]
    return df[["GAME_DATE", "PTS", "REB", "AST"]]

if page == "Home":
    st.subheader("Welcome 👋")
    st.write("Use the sidebar to compare player stats (PTS, REB, AST).")

elif page == "Compare Players":
    st.subheader("📊 Compare Two Players")

    # Auto-complete search
    player1_name = st.sidebar.selectbox("Player 1", player_names, index=player_names.index("LeBron James"))
    player2_name = st.sidebar.selectbox("Player 2", player_names, index=player_names.index("Kevin Durant"))

    # Fetch stats
    df1 = get_player_stats(player1_name, season)
    df2 = get_player_stats(player2_name, season)

    if df1 is None or df2 is None or df1.empty or df2.empty:
        st.error("❌ Could not fetch stats for one or both players.")
    else:
        # Chart toggle
        view = st.radio("Chart View", ["Side by Side", "Overlay"], horizontal=True)

        if view == "Side by Side":
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(df1.set_index("GAME_DATE")[["PTS", "REB", "AST"]])
                st.caption(f"📈 {player1_name}")
            with col2:
                st.line_chart(df2.set_index("GAME_DATE")[["PTS", "REB", "AST"]])
                st.caption(f"📈 {player2_name}")

        else:
            # Reshape data for Altair overlay
            df1_clean = df1.melt("GAME_DATE", var_name="Stat", value_name="Value")
            df1_clean["Player"] = player1_name

            df2_clean = df2.melt("GAME_DATE", var_name="Stat", value_name="Value")
            df2_clean["Player"] = player2_name

            combined_df = pd.concat([df1_clean, df2_clean])

            chart = (
                alt.Chart(combined_df)
                .mark_line(point=True)
                .encode(
                    x="GAME_DATE:T",
                    y="Value:Q",
                    color="Player:N",
                    strokeDash="Stat:N",  # Different line style for PTS/REB/AST
                    tooltip=["GAME_DATE:T", "Player", "Stat", "Value"]
                )
                .properties(width=900, height=450)
            )

            st.altair_chart(chart, use_container_width=True)
