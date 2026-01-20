import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Road Accident Analytics",
    page_icon="🚦",
    layout="wide"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned/US_Accidents_cleaned_sample_milestone1.csv")
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

    df["Date"] = df["Start_Time"].dt.date
    df["Hour"] = df["Start_Time"].dt.hour
    df["Month"] = df["Start_Time"].dt.month
    df["Day"] = df["Start_Time"].dt.day_name()
    return df

df = load_data()

# ✅ Reduce memory for faster filtering
df["State"] = df["State"].astype("category")
df["City"] = df["City"].astype("category")
df["Weather_Condition"] = df["Weather_Condition"].astype("category")
df["Day"] = df["Day"].astype("category")

# ================= TITLE =================
st.title("🚦 Road Accident Analytics Dashboard")
st.markdown("### 📊 Advanced Exploratory Data Analysis & Visual Insights")

# ================= SIDEBAR FILTERS (AUTO APPLY) =================
st.sidebar.header("🔎 Filters")

all_states = sorted(df["State"].dropna().unique())
all_severity = sorted(df["Severity"].unique())
all_weather = sorted(df["Weather_Condition"].dropna().unique())

min_date = df["Date"].min()
max_date = df["Date"].max()

state = st.sidebar.multiselect(
    "State",
    all_states,
    default=all_states[:10]  # ✅ 10 states default
)

severity = st.sidebar.multiselect(
    "Severity Level",
    all_severity
)

weather = st.sidebar.multiselect(
    "Weather Condition",
    all_weather
)

day = st.sidebar.multiselect(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

date_range = st.sidebar.date_input(
    "Select Date Range 📅",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

hour_range = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))
month_range = st.sidebar.slider("Month", 1, 12, (1, 12))

# Date safety check
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# ================= APPLY FILTERS (CACHED) =================
@st.cache_data
def get_filtered_data(df, state, severity, weather, day, hour_range, month_range, start_date, end_date):
    filtered_df = df.copy()

    # Date filter
    filtered_df = filtered_df[
        (filtered_df["Date"] >= start_date) &
        (filtered_df["Date"] <= end_date)
    ]

    if state:
        filtered_df = filtered_df[filtered_df["State"].isin(state)]
    if severity:
        filtered_df = filtered_df[filtered_df["Severity"].isin(severity)]
    if weather:
        filtered_df = filtered_df[filtered_df["Weather_Condition"].isin(weather)]
    if day:
        filtered_df = filtered_df[filtered_df["Day"].isin(day)]

    filtered_df = filtered_df[
        (filtered_df["Hour"].between(hour_range[0], hour_range[1])) &
        (filtered_df["Month"].between(month_range[0], month_range[1]))
    ]

    return filtered_df

filtered_df = get_filtered_data(df, state, severity, weather, day, hour_range, month_range, start_date, end_date)

# ================= METRICS =================
st.subheader("📌 Key Metrics")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Accidents", f"{len(filtered_df):,}")
c2.metric("Average Severity", round(filtered_df["Severity"].mean(), 2) if len(filtered_df) > 0 else 0)
c3.metric("States Covered", filtered_df["State"].nunique())
c4.metric("Cities Covered", filtered_df["City"].nunique())
c5.metric("Weather Types", filtered_df["Weather_Condition"].nunique())

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 EDA Analysis", "🗺️ Accident Maps", "🌦 Weather Analysis", "📊 Correlation", "📌 Help"]
)

# ================= TAB 1 : EDA =================
with tab1:
    st.subheader("Severity Distribution")
    fig1 = px.histogram(
        filtered_df,
        x="Severity",
        color="Severity",
        template="plotly_dark"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Accidents by Hour")
    fig2 = px.line(
        filtered_df.groupby("Hour").size().reset_index(name="Count"),
        x="Hour",
        y="Count",
        markers=True,
        template="plotly_dark"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Accidents by Day of Week")
    fig3 = px.bar(
        filtered_df.groupby("Day").size().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ).reset_index(name="Count"),
        x="Day",
        y="Count",
        template="plotly_dark"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Accidents by Month")
    fig4 = px.bar(
        filtered_df.groupby("Month").size().reset_index(name="Count"),
        x="Month",
        y="Count",
        template="plotly_dark"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Top 15 Accident-Prone States")
    state_counts = (
        filtered_df.groupby("State")
        .size()
        .reset_index(name="Accident Count")
        .sort_values("Accident Count", ascending=False)
        .head(15)
    )

    fig5 = px.bar(
        state_counts,
        x="State",
        y="Accident Count",
        color="Accident Count",
        template="plotly_dark"
    )
    st.plotly_chart(fig5, use_container_width=True)

# ================= TAB 2 : MAPS (OPTIMIZED) =================
with tab2:
    st.subheader("🗺️ Accident Maps (Optimized)")

    # ✅ Reduce lag by limiting map points
    map_points = st.slider("Map Points (Reduce for faster loading)", 500, 4000, 1500, step=500)

    map_view = st.radio(
        "Map View",
        ["Hotspots Heatmap", "Marker Clusters"],
        horizontal=True
    )

    map_df = filtered_df.dropna(subset=["Start_Lat", "Start_Lng"])

    if len(map_df) > map_points:
        map_df = map_df.sample(n=map_points, random_state=42)

    if len(map_df) == 0:
        st.warning("No map data available for selected filters.")
    else:
        if map_view == "Hotspots Heatmap":
            st.subheader("🔥 Accident Hotspots Heatmap")

            heat_map = folium.Map(
                location=[map_df["Start_Lat"].mean(), map_df["Start_Lng"].mean()],
                zoom_start=5,
                tiles="CartoDB dark_matter"
            )

            HeatMap(
                map_df[["Start_Lat", "Start_Lng"]].values.tolist(),
                radius=12,
                blur=15
            ).add_to(heat_map)

            st_folium(heat_map, height=550, width="100%")

        else:
            st.subheader("📍 Marker Cluster Map")

            m = folium.Map(
                location=[map_df["Start_Lat"].mean(), map_df["Start_Lng"].mean()],
                zoom_start=5,
                tiles="CartoDB positron"
            )

            cluster = MarkerCluster().add_to(m)
            for _, row in map_df.iterrows():
                folium.CircleMarker(
                    location=[row["Start_Lat"], row["Start_Lng"]],
                    radius=3,
                    color="red",
                    fill=True,
                    fill_opacity=0.6
                ).add_to(cluster)

            st_folium(m, height=550, width="100%")

# ================= TAB 3 : WEATHER =================
with tab3:
    st.subheader("Weather Condition Distribution")
    fig6 = px.bar(
        filtered_df["Weather_Condition"].value_counts().head(15),
        template="plotly_dark"
    )
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Severity vs Weather")
    fig7 = px.box(
        filtered_df,
        x="Weather_Condition",
        y="Severity",
        template="plotly_dark"
    )
    st.plotly_chart(fig7, use_container_width=True)

# ================= TAB 4 : CORRELATION =================
with tab4:
    st.subheader("Correlation Heatmap")

    corr_cols = [
        "Severity",
        "Visibility(mi)",
        "Temperature(F)",
        "Wind_Speed(mph)"
    ]

    corr_df = filtered_df[corr_cols].dropna()

    if len(corr_df) == 0:
        st.warning("Not enough data available for correlation heatmap.")
    else:
        corr = corr_df.corr()

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ================= TAB 5 : HELP =================
with tab5:
    st.subheader("📌 How to Use This Dashboard (Guidelines)")

    st.markdown("""
✅ **Step 1: Select Filters (Left Sidebar)**  
- Choose **State**, **Severity**, **Weather**, and **Day of Week**  
- Select **Date Range**, **Hour Range**, and **Month Range**  
- Dashboard updates automatically after selection  

✅ **Step 2: View Key Metrics**  
- Total accidents, average severity, and coverage details are shown at the top  

✅ **Step 3: Explore Tabs**  
📈 **EDA Analysis** → Trends by hour, day, month, and severity  
🗺️ **Accident Maps** → Hotspots heatmap + marker clusters  
🌦 **Weather Analysis** → Weather distribution & severity comparison  
📊 **Correlation** → Relationship between severity & weather numeric features  

✅ **Tip for Faster Maps**  
- Reduce the **Map Points slider** to improve performance.
""")

# ================= FOOTER =================
st.markdown("---")
st.markdown("🚀 **Developed by Keerthi | Advanced Road Safety Analytics**")
