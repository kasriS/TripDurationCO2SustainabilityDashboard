# src/eda.py
# Enhanced EDA module with comprehensive visualizations
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns

NYC_VIEW_STATE = pdk.ViewState(latitude=40.758, longitude=-73.9855, zoom=11, pitch=45)

def prepare_for_eda(df):
    # Enhanced preparation with more derived features
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    # Compute haversine distance if coordinates are available
    if {"pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"}.issubset(df.columns):
        df["haversine_dist"] = haversine_series(df["pickup_latitude"], df["pickup_longitude"],
                                                df["dropoff_latitude"], df["dropoff_longitude"])
    else:
        df["haversine_dist"] = 0.0

    # Extract temporal features
    df["hour"] = df["pickup_datetime"].dt.hour.fillna(0).astype(int)
    df["weekday"] = df["pickup_datetime"].dt.weekday.fillna(0).astype(int)
    df["month"] = df["pickup_datetime"].dt.month.fillna(0).astype(int)

    # Add season mapping
    df["season"] = df["month"].apply(
        lambda x: 'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else 'Fall'
    )

    # Add log trip duration if available
    if 'trip_duration' in df.columns:
        df['log_trip_duration'] = np.log1p(df['trip_duration'])

    return df

def haversine_series(lat1, lon1, lat2, lon2):
    # Vectorized haversine calculation
    R = 6371.0
    lat1r = np.radians(lat1.to_numpy(dtype=float))
    lon1r = np.radians(lon1.to_numpy(dtype=float))
    lat2r = np.radians(lat2.to_numpy(dtype=float))
    lon2r = np.radians(lon2.to_numpy(dtype=float))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def render_eda():
    st.title("🔍 Comprehensive NYC Taxi Trip Analysis")
    st.write(
        "Upload a sample trips file to explore distributions. "
        "Accepted formats: CSV or Parquet with required columns: "
        "`pickup_datetime, pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, passenger_count`."
    )
    st.markdown("---")

    # Initialize df to None at the start
    df = None
    up = st.file_uploader("Upload trips (CSV/Parquet) for EDA", type=["csv", "parquet"])
    use_kaggle = st.checkbox(
        "Attempt to load Kaggle sample at /kaggle/input/nyc-taxi-trip-duration/train.zip (optional)")

    # Handle all data loading scenarios
    if up is not None:
        try:
            if up.name.endswith(".csv"):
                df = pd.read_csv(up)
            else:
                df = pd.read_parquet(up)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            return
    elif use_kaggle:
        kag_path = "/kaggle/input/nyc-taxi-trip-duration/train.zip"
        if os.path.exists(kag_path):
            try:
                st.info("Loading Kaggle sample...")
                df = pd.read_csv(kag_path)
            except Exception as e:
                st.warning(f"Could not load Kaggle sample: {e}")
        else:
            st.warning("Kaggle sample path not found in this environment.")

    # If df is still None, use synthetic data
    if df is None:
        st.info("No file uploaded — using synthetic sample.")
        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame({
            "pickup_datetime": pd.date_range("2016-01-01", periods=n, freq="30min"),
            "pickup_latitude": rng.uniform(40.6, 40.85, n),
            "pickup_longitude": rng.uniform(-74.05, -73.75, n),
            "dropoff_latitude": rng.uniform(40.6, 40.85, n),
            "dropoff_longitude": rng.uniform(-74.05, -73.75, n),
            "passenger_count": rng.integers(1, 5, n),
            "trip_duration": rng.integers(300, 3600, n)  # Add duration for better EDA
        })

    # Prepare data for EDA
    df = prepare_for_eda(df)

    # Set consistent plotting style
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.size'] = 12

    # =====================
    # Data Quality Analysis
    # =====================
    st.header("📊 Data Quality Analysis")

    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        missing_data = pd.DataFrame(missing_counts, columns=['Missing Count'])
        st.dataframe(missing_data)
    else:
        st.success("✅ No missing values found in the dataset")
    st.markdown("---")

    # =====================
    # Trip Duration Analysis
    # =====================
    if 'trip_duration' in df.columns:
        st.header("⏱ Trip Duration Analysis")

        # Log-transformed distribution
        st.subheader("Trip Duration Distribution (Log Scale)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['log_trip_duration'], bins=50, kde=True, color='purple')
        plt.title('Log-Transformed Trip Duration Distribution')
        plt.xlabel('Log(Trip Duration)')
        plt.ylabel('Count')
        st.pyplot(fig)
    else:
        st.warning("No 'trip_duration' column found - duration-related analysis skipped")
    st.markdown("---")

    # =====================
    # Temporal Analysis
    # =====================
    st.header("📅 Temporal Patterns")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trips by Hour")
        hour_counts = df.groupby("hour").size().reset_index(name="trips")
        fig = px.bar(hour_counts, x="hour", y="trips", color="trips",
                     color_continuous_scale='bluered')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Trips by Weekday")
        wd = df.groupby("weekday").size().reset_index(name="trips")
        fig = px.bar(wd, x="weekday", y="trips", color="trips",
                     color_continuous_scale='tealrose')
        st.plotly_chart(fig, use_container_width=True)

    # Hourly duration pattern
    if 'log_trip_duration' in df.columns:
        st.subheader("Average Trip Duration by Hour")
        fig, ax = plt.subplots(figsize=(14, 6))
        hourly_avg = df.groupby('hour')['log_trip_duration'].mean()
        sns.lineplot(x=hourly_avg.index, y=hourly_avg.values,
                     color='royalblue', linewidth=2.5)
        plt.title('Average Trip Duration by Hour of Day', pad=20)
        plt.xlabel('Hour of Day', labelpad=10)
        plt.ylabel('Log(Trip Duration)', labelpad=10)
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Weekly duration pattern
    if 'log_trip_duration' in df.columns:
        st.subheader("Average Trip Duration by Day of Week")
        fig, ax = plt.subplots(figsize=(14, 6))
        weekday_avg = df.groupby('weekday')['log_trip_duration'].mean()
        sns.lineplot(x=weekday_avg.index, y=weekday_avg.values,
                     color='crimson', linewidth=2.5)
        plt.title('Average Trip Duration by Day of Week', pad=20)
        plt.xlabel('Day of Week', labelpad=10)
        plt.ylabel('Log(Trip Duration)', labelpad=10)
        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Seasonal pattern
    if 'log_trip_duration' in df.columns:
        st.subheader("Seasonal Patterns")
        fig, ax = plt.subplots(figsize=(14, 6))
        seasonal_avg = df.groupby(['month', 'season'])['log_trip_duration'].mean().reset_index()
        palette = {'Winter': 'steelblue', 'Spring': 'forestgreen',
                   'Summer': 'goldenrod', 'Fall': 'firebrick'}
        sns.lineplot(x='month', y='log_trip_duration', hue='season',
                     data=seasonal_avg, palette=palette,
                     linewidth=2.5, marker='o', markersize=8)
        plt.title('Average Trip Duration by Month & Season', pad=20)
        plt.xlabel('Month', labelpad=10)
        plt.ylabel('Log(Trip Duration)', labelpad=10)
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True, alpha=0.3)
        plt.legend(title='Season')
        st.pyplot(fig)
    st.markdown("---")

    # =====================
    # Passenger Analysis
    # =====================
    if 'passenger_count' in df.columns:
        st.header("👥 Passenger Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Passenger Count Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='passenger_count', data=df, palette='viridis')
            plt.title('Distribution of Passenger Counts')
            plt.xlabel('Number of Passengers')
            plt.ylabel('Count')
            st.pyplot(fig)

        with col2:
            if 'trip_duration' in df.columns:
                st.subheader("Duration by Passenger Count")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(x='passenger_count', y='trip_duration', data=df, palette='coolwarm')
                plt.yscale('log')
                plt.title('Trip Duration by Passenger Count')
                plt.xlabel('Passenger Count')
                plt.ylabel('Trip Duration (log scale)')
                st.pyplot(fig)
    st.markdown("---")

    # =====================
    # Geographic Analysis
    # =====================
    if {'pickup_longitude', 'pickup_latitude'}.issubset(df.columns):
        st.header("🗺️ Geographic Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Pickup Heatmap")
            fig = px.density_heatmap(
                df,
                x='pickup_longitude',
                y='pickup_latitude',
                nbinsx=40,
                nbinsy=40,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Pickup Locations")
            fig, ax = plt.subplots(figsize=(10, 8))
            sample_size = min(10000, len(df))
            sns.scatterplot(
                x='pickup_longitude',
                y='pickup_latitude',
                data=df.sample(sample_size),
                alpha=0.1,
                color='red'
            )
            plt.title('Pickup Locations in NYC')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.xlim(-74.05, -73.75)
            plt.ylim(40.60, 40.90)
            st.pyplot(fig)

        st.subheader("Interactive Map Preview")
        sample = df.sample(min(1000, len(df)), random_state=42)
        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=sample.rename(columns={"pickup_longitude": "lon", "pickup_latitude": "lat"}),
                get_position='[lon, lat]',
                get_radius=40,
                get_fill_color=[0, 128, 255, 120]
            )
        ]
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v10",
                initial_view_state=NYC_VIEW_STATE,
                layers=layers
            )
        )
    st.markdown("---")

    # =====================
    # Distance Analysis
    # =====================
    if 'haversine_dist' in df.columns and df['haversine_dist'].sum() > 0:
        st.header("📏 Distance Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distance Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['haversine_dist'], bins=50, kde=True, color='green')
            plt.title('Haversine Distance Distribution')
            plt.xlabel('Distance (km)')
            plt.ylabel('Count')
            st.pyplot(fig)

        with col2:
            if 'trip_duration' in df.columns:
                st.subheader("Duration vs Distance")
                fig, ax = plt.subplots(figsize=(10, 6))
                sample_size = min(1000, len(df))
                sns.scatterplot(
                    x='haversine_dist',
                    y='trip_duration',
                    data=df.sample(sample_size),
                    alpha=0.6
                )
                plt.title('Trip Duration vs Distance')
                plt.xlabel('Distance (km)')
                plt.ylabel('Trip Duration (sec)')
                plt.yscale('log')
                st.pyplot(fig)

        if 'trip_duration' in df.columns:
            st.subheader("Log Duration vs Distance")
            fig, ax = plt.subplots(figsize=(12, 6))
            sample_size = min(1000, len(df))
            sns.scatterplot(
                x='haversine_dist',
                y='log_trip_duration',
                data=df.sample(sample_size),
                alpha=0.6,
                hue=df['passenger_count'] if 'passenger_count' in df.columns else None,
                palette='viridis'
            )
            plt.title('Log Trip Duration vs Distance')
            plt.xlabel('Distance (km)')
            plt.ylabel('Log(Trip Duration)')
            st.pyplot(fig)
    st.markdown("---")

    # =====================
    # Correlation Analysis
    # =====================
    st.header("📈 Correlation Analysis")

    numerical_features = [
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'passenger_count', 'haversine_dist'
    ]

    # Add trip duration features if available
    if 'trip_duration' in df.columns:
        numerical_features.extend(['trip_duration', 'log_trip_duration'])

    # Filter to existing features only
    numerical_features = [f for f in numerical_features if f in df.columns]

    if len(numerical_features) > 1:
        st.subheader("Correlation Matrix")
        corr_matrix = df[numerical_features].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            ax=ax,
            linewidths=0.5
        )
        plt.title('Feature Correlation Matrix')
        st.pyplot(fig)
    else:
        st.warning("Insufficient numerical features for correlation analysis")


if __name__ == "__main__":
    render_eda()