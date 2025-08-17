import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ==========================================================
#  Utility Functions
# ==========================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two lat/lon points in km."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def prepare_data(df):
    """Ensure numeric lat/lon and drop missing, convert datetime if exists."""
    needed = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

    # Check for datetime columns
    datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]

    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Dataset must contain {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert datetime columns if found
    for col in datetime_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            continue

    return df.dropna(subset=needed)


# ==========================================================
#  Dashboard Logic
# ==========================================================

EMISSION_FACTOR = 0.120  # kg CO₂ per km


def render_routes_scenarios(trips_df, osrm_df=None):
    st.header("Route Optimization Scenarios")

    # --- User input ---
    reduction_pct = st.slider("Average Route Reduction (%)", 0, 20, 6)
    base_avg_km = 5.8  # fallback if distances missing

    # --- Prepare data ---
    df_prep = prepare_data(trips_df)

    # --- Distance calculation ---
    if osrm_df is not None and "osrm_total_distance_km" in osrm_df.columns:
        df_merged = df_prep.merge(osrm_df, on="trip_id", how="left")
        orig_dist_km = df_merged["osrm_total_distance_km"].fillna(
            df_merged.apply(
                lambda r: haversine_km(
                    r["pickup_latitude"], r["pickup_longitude"],
                    r["dropoff_latitude"], r["dropoff_longitude"]
                ), axis=1
            )
        ).to_numpy(dtype=float)
    else:
        orig_dist_km = df_prep.apply(
            lambda r: haversine_km(
                r["pickup_latitude"], r["pickup_longitude"],
                r["dropoff_latitude"], r["dropoff_longitude"]
            ), axis=1
        ).fillna(base_avg_km).to_numpy(dtype=float)

    # --- Apply reduction ---
    factor = (100.0 - reduction_pct) / 100.0
    optimized_km = orig_dist_km * factor

    # --- CO₂ and Distance savings ---
    dist_saved_each = orig_dist_km - optimized_km
    co2_saved_each = dist_saved_each * EMISSION_FACTOR

    co2_saved_total = float(np.nansum(co2_saved_each))
    dist_saved_total = float(np.nansum(dist_saved_each))

    # --- Dashboard cards ---
    st.subheader("Scenario Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trips", f"{len(df_prep):,}")
    col2.metric("Avg reduction", f"{reduction_pct}%")
    col3.metric("Total CO₂ Saved", f"{co2_saved_total:,.2f} kg")
    col4.metric("Total Distance Saved", f"{dist_saved_total:,.2f} km")

    # --- Show sample trips ---
    st.subheader("Sample of Trips with Savings")
    sample = df_prep.head(100).copy()
    sample["Original_km"] = orig_dist_km[:100]
    sample["Optimized_km"] = optimized_km[:100]
    sample["Distance_Saved_km"] = dist_saved_each[:100]
    sample["CO2_Saved_kg"] = co2_saved_each[:100]
    st.dataframe(sample[[
        "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude",
        "Original_km", "Optimized_km", "Distance_Saved_km", "CO2_Saved_kg"
    ]])

    # ==========================================================
    # 🔽 NEW: Daily CO2 Savings Breakdown
    # ==========================================================
    st.divider()
    st.subheader("Daily CO₂ Savings Breakdown")

    # Find datetime columns for aggregation
    datetime_cols = [col for col in df_prep.columns
                     if isinstance(df_prep[col].dtype, pd.DatetimeTZDtype) or
                     pd.api.types.is_datetime64_any_dtype(df_prep[col])]

    if datetime_cols:
        date_col = st.selectbox("Select date column for daily aggregation",
                                datetime_cols,
                                index=0)

        # Create daily dataframe
        daily_df = df_prep.copy()
        daily_df['date'] = pd.to_datetime(daily_df[date_col]).dt.date
        daily_df['original_co2'] = orig_dist_km * EMISSION_FACTOR
        daily_df['optimized_co2'] = optimized_km * EMISSION_FACTOR
        daily_df['co2_saved'] = co2_saved_each

        # Aggregate by date
        daily_summary = daily_df.groupby('date').agg({
            'original_co2': 'sum',
            'optimized_co2': 'sum',
            'co2_saved': 'sum'
        }).reset_index()

        # Sort by date
        daily_summary = daily_summary.sort_values('date')

        # Show daily savings table
        st.dataframe(daily_summary.style.format({
            'original_co2': '{:,.2f}',
            'optimized_co2': '{:,.2f}',
            'co2_saved': '{:,.2f}'
        }))

        # ==========================================================
        # 🔽 NEW: Before/After Optimization Line Plot
        # ==========================================================
        st.subheader("Daily CO₂ Emissions: Before vs After Optimization")

        # Create line plot
        fig = go.Figure()

        # Original emissions line
        fig.add_trace(go.Scatter(
            x=daily_summary['date'],
            y=daily_summary['original_co2'],
            mode='lines+markers',
            name='Original CO₂',
            line=dict(color='red', width=3),
            marker=dict(size=8, color='red')
        ))

        # Optimized emissions line
        fig.add_trace(go.Scatter(
            x=daily_summary['date'],
            y=daily_summary['optimized_co2'],
            mode='lines+markers',
            name='Optimized CO₂',
            line=dict(color='green', width=3, dash='dash'),
            marker=dict(size=8, color='green')
        ))

        # Savings area
        fig.add_trace(go.Scatter(
            x=daily_summary['date'],
            y=daily_summary['optimized_co2'],
            mode='none',
            name='CO₂ Savings',
            fill='tonexty',
            fillcolor='rgba(100, 200, 100, 0.2)'
        ))

        # Layout settings
        fig.update_layout(
            title='Daily CO₂ Emissions Reduction',
            xaxis_title='Date',
            yaxis_title='CO₂ Emissions (kg)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # ==========================================================
        # 🔽 NEW: CO₂ Savings Trend
        # ==========================================================
        st.subheader("Daily CO₂ Savings Trend")

        fig_savings = px.area(
            daily_summary,
            x='date',
            y='co2_saved',
            title='Daily CO₂ Savings',
            labels={'co2_saved': 'CO₂ Saved (kg)'},
            color_discrete_sequence=['#2ca02c']
        )

        fig_savings.update_layout(
            hovermode='x',
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_savings, use_container_width=True)
    else:
        st.warning("No datetime columns found for daily aggregation. Add a datetime column to enable daily breakdowns.")

    # --- Scenario Comparison ---
    st.divider()
    st.markdown("### Scenario Comparison: Optimization Levels")

    scenario_data = []
    for pct in [0, 6, 12, 20]:
        factor = (100.0 - pct) / 100.0
        optimized_km_scenario = orig_dist_km * factor
        dist_saved = orig_dist_km - optimized_km_scenario
        scenario_data.append({
            "Optimization %": pct,
            "CO₂ Saved (kg)": (dist_saved * EMISSION_FACTOR).sum(),
            "Distance Saved (km)": dist_saved.sum()
        })

    scenario_df = pd.DataFrame(scenario_data)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CO₂ Savings by Optimization Level")
        fig_co2 = px.bar(
            scenario_df, x="Optimization %", y="CO₂ Saved (kg)",
            color="CO₂ Saved (kg)", color_continuous_scale="tealrose"
        )
        st.plotly_chart(fig_co2, use_container_width=True)

    with col2:
        st.subheader("Distance Reduction by Optimization Level")
        fig_dist = px.bar(
            scenario_df, x="Optimization %", y="Distance Saved (km)",
            color="Distance Saved (km)", color_continuous_scale="bluered"
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# ==========================================================
#  Main App
# ==========================================================

def main():
    st.title("NYC Taxi – Route Optimization Dashboard")

    st.info("Upload the main trip dataset and optionally the OSRM fastest routes dataset.")

    trips_file = st.file_uploader("Upload trips dataset (dataset/data.csv)", type=["csv"])
    osrm_file = st.file_uploader("Upload OSRM distances (dataset/fastestRoutes.csv)", type=["csv"])

    if trips_file is not None:
        trips_df = pd.read_csv(trips_file)
        osrm_df = pd.read_csv(osrm_file) if osrm_file is not None else None

        # Add button to trigger calculations
        if st.button("Calculate Optimization Scenarios", type="primary"):
            with st.spinner("Calculating routes and emissions..."):
                render_routes_scenarios(trips_df, osrm_df)
    else:
        st.warning("Please upload the trips dataset to proceed.")


if __name__ == "__main__":
    main()
