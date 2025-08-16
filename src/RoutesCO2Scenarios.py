# src/routes_scenarios_page.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from features import prepare_data
from predict import estimate_co2_saving

def number_card(label, value, suffix=""):
    st.metric(label, f"{value}{suffix}")

def section_title(txt):
    st.markdown(f"### {txt}")

def render_routes_scenarios():
    st.title("🗺️ Route Optimization — Multi-trip Scenarios")
    st.write(
        "Upload a trips file or generate synthetic data. Choose a scenario window (3-day or monthly), "
        "set an **optimization improvement** (distance reduction in %), and visualize total **CO₂ savings**."
    )

    up = st.file_uploader("Trips file (CSV/Parquet)", type=["csv", "parquet"])
    scenario = st.selectbox("Scenario window", ["3-day", "Monthly"])
    trips_target = st.number_input("If no file: number of synthetic trips", min_value=200, max_value=20000, value=2200, step=100)
    reduction_pct = st.slider("Average route distance reduction (%)", min_value=0, max_value=30, value=6, step=1)
    base_avg_km = st.slider("Assumed original avg distance per trip (km)", min_value=1.0, max_value=20.0, value=5.8, step=0.1)

    if st.button("Run Scenario"):
        if up is not None:
            if up.name.endswith(".csv"):
                df = pd.read_csv(up)
            else:
                df = pd.read_parquet(up)
        else:
            rng = np.random.default_rng(123)
            start = pd.Timestamp("2016-01-01")
            if scenario == "3-day":
                dates = pd.date_range(start, periods=3*24*2, freq="30min")
            else:
                dates = pd.date_range(start, periods=30*24, freq="1h")

            n = trips_target
            df = pd.DataFrame({
                "pickup_datetime": rng.choice(dates, size=n, replace=True),
                "pickup_latitude": rng.uniform(40.6, 40.85, n),
                "pickup_longitude": rng.uniform(-74.05, -73.75, n),
                "dropoff_latitude": rng.uniform(40.6, 40.85, n),
                "dropoff_longitude": rng.uniform(-74.05, -73.75, n),
                "passenger_count": rng.integers(1, 5, n),
            })

        df_prep = prepare_data(df, fit_kmeans=False)
        if "osrm_total_distance_km" in df_prep.columns and df_prep["osrm_total_distance_km"].notna().any():
            orig_dist_km = df_prep["osrm_total_distance_km"].fillna(base_avg_km).to_numpy()
        else:
            orig_dist_km = np.full(len(df_prep), base_avg_km)

        factor = (100.0 - reduction_pct) / 100.0
        optimized_km = orig_dist_km * factor

        co2_saved_each = estimate_co2_saving(orig_dist_km, optimized_km)
        co2_saved_series = pd.Series(co2_saved_each if not np.isscalar(co2_saved_each) else np.full(len(df_prep), co2_saved_each))
        co2_saved_total = co2_saved_series.sum()

        df_plot = pd.DataFrame({
            "pickup_datetime": pd.to_datetime(df_prep["pickup_datetime"], errors="coerce"),
            "orig_km": orig_dist_km,
            "opt_km": optimized_km,
            "co2_saved": co2_saved_series.values
        })
        df_plot["date"] = df_plot["pickup_datetime"].dt.date
        daily = df_plot.groupby("date").agg(
            trips=("orig_km", "size"),
            total_km=("orig_km", "sum"),
            total_km_opt=("opt_km", "sum"),
            co2_saved=("co2_saved", "sum"),
        ).reset_index()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            number_card("Trips", len(df_prep))
        with c2:
            number_card("Avg reduction", f"{reduction_pct}", "%")
        with c3:
            number_card("Total CO₂ Saved", f"{co2_saved_total:.1f}", " kg")
        with c4:
            km_saved = (df_plot["orig_km"].sum() - df_plot["opt_km"].sum())
            number_card("Total Distance Saved", f"{km_saved:.1f}", " km")

        st.divider()
        section_title("Daily CO₂ Savings")
        fig = px.bar(daily, x="date", y="co2_saved", labels={"co2_saved":"kg CO₂"})
        st.plotly_chart(fig, use_container_width=True)

        section_title("Original vs Optimized Distance — by Day")
        melted = daily.melt(id_vars="date", value_vars=["total_km", "total_km_opt"], var_name="type", value_name="km")
        fig2 = px.line(melted, x="date", y="km", color="type")
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "These results assume a uniform average trip distance if no OSRM distance is available, "
            "and apply a global reduction percentage. Plug in your real routing outputs to refine."
        )
