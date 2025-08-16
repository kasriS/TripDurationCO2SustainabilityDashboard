
# src/dashboard.py
import math
import pandas as pd
import streamlit as st
import pydeck as pdk
from geopy.geocoders import Nominatim

from features import prepare_data
from predict import load_artifacts, predict_duration, estimate_co2_saving, haversine_km
import eda
from RoutesCO2Scenarios import render_routes_scenarios  # imported separately

# Default NYC view
NYC_VIEW_STATE = pdk.ViewState(latitude=40.758, longitude=-73.9855, zoom=11, pitch=45)

# Utilities
def straight_line_path(pickup, dropoff, wiggle=False):
    lon1, lat1 = pickup
    lon2, lat2 = dropoff
    if not wiggle:
        return [[lon1, lat1], [lon2, lat2]]
    mid_lon = (lon1 + lon2) / 2
    mid_lat = (lat1 + lat2) / 2
    dx = lon2 - lon1
    dy = lat2 - lat1
    norm = math.hypot(dx, dy) or 1.0
    off = 0.008
    off_lon = -dy / norm * off
    off_lat = dx / norm * off
    return [[lon1, lat1], [mid_lon + off_lon, mid_lat + off_lat], [lon2, lat2]]

def number_card(label, value, suffix=""):
    st.metric(label, f"{value}{suffix}")

def section_title(txt):
    st.markdown(f"### {txt}")

# Home page
def page_home():
    st.title("🚕 NYC Taxi • Trip Duration & CO₂ Sustainability")
    st.write(
        "Welcome! Use the sidebar to explore **EDA**, run **single-trip predictions**, or evaluate **route optimization scenarios**."
    )

    col1, col2, col3 = st.columns(3)
    number_card("Sample Predicted Duration (min)", 12.4)
    number_card("CO₂ Saved (kg) — Example", 0.37)
    number_card("Monthly Scenario Trips", "2,200")

    st.divider()
    section_title("Quick Map Preview")

    pickup = {"name": "Pickup", "lat": 40.758, "lon": -73.9855}
    dropoff = {"name": "Dropoff", "lat": 40.775, "lon": -73.957}
    path = straight_line_path((pickup["lon"], pickup["lat"]), (dropoff["lon"], dropoff["lat"]), wiggle=True)

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=[pickup, dropoff],
            get_position='[lon, lat]',
            get_radius=60,
            get_fill_color=[0, 128, 255, 200],
        ),
        pdk.Layer(
            "PathLayer",
            data=[{"path": path}],
            get_path="path",
            width_scale=4,
            width_min_pixels=2,
            get_width=3,
            get_color=[0, 0, 0, 200],
        ),
    ]
    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v10",
                             initial_view_state=NYC_VIEW_STATE, layers=layers))

# EDA page
def page_eda():
    eda.render_eda()

# Prediction page
def page_prediction():
    st.title("🔮 Single Trip Prediction + Map & CO₂")
    load_artifacts()
    geolocator = Nominatim(user_agent="nyc_taxi_dashboard")

    st.info("You can explore the map below to see landmark names for easier address entry.")

    # Example landmark data for map (add more landmarks if needed)
    landmarks = pd.DataFrame([
        {"name": "Times Square", "lat": 40.7580, "lon": -73.9855},
        {"name": "Central Park", "lat": 40.7851, "lon": -73.9683},
        {"name": "Empire State Building", "lat": 40.7484, "lon": -73.9857},
        {"name": "Brooklyn Bridge", "lat": 40.7061, "lon": -73.9969},
        {"name": "Statue of Liberty", "lat": 40.6892, "lon": -74.0445},
    ])

    # Map visualization
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state=NYC_VIEW_STATE,
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=landmarks,
                get_position='[lon, lat]',
                get_radius=100,
                get_fill_color=[255, 0, 0, 200],
                pickable=True,
                auto_highlight=True,
            )
        ],
        tooltip={"text": "{name}\nLat: {lat}\nLon: {lon}"}
    ))

    # User form
    with st.form("single_trip"):
        pickup_address = st.text_input("Pickup Address / Landmark", "Times Square, NYC")
        drop_address = st.text_input("Dropoff Address / Landmark", "Central Park, NYC")

        # Geocode addresses
        pickup_location = geolocator.geocode(pickup_address)
        drop_location = geolocator.geocode(drop_address)

        if pickup_location and drop_location:
            pickup_lat, pickup_lon = pickup_location.latitude, pickup_location.longitude
            drop_lat, drop_lon = drop_location.latitude, drop_location.longitude
            st.success(f"Pickup: ({pickup_lat:.6f}, {pickup_lon:.6f}), Dropoff: ({drop_lat:.6f}, {drop_lon:.6f})")
        else:
            st.error("Could not geocode one of the addresses. Please try a different address.")
            return

        pickup_dt = st.text_input("Pickup Datetime (YYYY-MM-DD HH:MM:SS)", "2016-01-01 08:30:00")
        passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
        orig_distance_km = st.number_input("Original Route Distance (km)", value=5.0)
        optim_distance_km = st.number_input("Optimized Route Distance (km)", value=4.7)
        show_wiggly = st.checkbox("Show curved path", True)
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    # Prepare data and predict
    df = pd.DataFrame([{
        "id": "app_demo",
        "pickup_datetime": pickup_dt,
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": drop_lon,
        "dropoff_latitude": drop_lat,
        "passenger_count": passenger_count
    }])
    df_prep = prepare_data(df, fit_kmeans=False)
    duration_s = predict_duration(df_prep)
    hav_km = haversine_km(pickup_lat, pickup_lon, drop_lat, drop_lon)

    path_orig = straight_line_path((pickup_lon, pickup_lat), (drop_lon, drop_lat), wiggle=False)
    path_opt = straight_line_path((pickup_lon, pickup_lat), (drop_lon, drop_lat), wiggle=show_wiggly)

    pickup_point = {"name": "Pickup", "lat": pickup_lat, "lon": pickup_lon}
    dropoff_point = {"name": "Dropoff", "lat": drop_lat, "lon": drop_lon}

    # Show route map
    layers = [
        pdk.Layer("ScatterplotLayer", data=[pickup_point, dropoff_point],
                  get_position='[lon, lat]', get_radius=70, get_fill_color=[0,128,255,200]),
        pdk.Layer("PathLayer", data=[{"path": path_orig}],
                  get_path="path", width_scale=4, width_min_pixels=2, get_width=3, get_color=[220,0,0,180]),
        pdk.Layer("PathLayer", data=[{"path": path_opt}],
                  get_path="path", width_scale=4, width_min_pixels=2, get_width=3, get_color=[0,140,0,200]),
    ]
    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v10",
                             initial_view_state=NYC_VIEW_STATE, layers=layers))

    # Show metrics
    c1, c2, c3 = st.columns(3)
    number_card("Predicted Duration", f"{duration_s/60:.2f}", " min")
    number_card("Straight-line Distance", f"{hav_km:.2f}", " km")
    co2_saved = estimate_co2_saving(orig_distance_km, optim_distance_km)
    number_card("CO₂ Saved", f"{co2_saved:.3f}", " kg")

# Model info page
def page_model_info():
    st.title("🧠 Model Information")
    load_artifacts()
    st.markdown("""
**Goal**: Predict taxi trip duration from engineered features and estimate CO₂ savings.
- Feature engineering: temporal, spatial, osrm optional, holidays/rush
- Models: XGBoost/LGBM/CatBoost stack typically
- Target: log1p(duration) -> invert with expm1
""")

# Main
def main():
    st.set_page_config(page_title="NYC Taxi — Dashboard", layout="wide", page_icon="🚕")
    with st.sidebar:
        st.header("🚕 Dashboard Menu")
        page = st.radio("Navigate", ["Home", "EDA", "Prediction", "Routes & CO₂ Scenarios", "Model Info"])
        st.caption("Tip: Add Kaggle-trained artifacts into `/models`.")

    if page == "Home":
        page_home()
    elif page == "EDA":
        page_eda()
    elif page == "Prediction":
        page_prediction()
    elif page == "Routes & CO₂ Scenarios":
        render_routes_scenarios()
    else:
        page_model_info()

if __name__ == "__main__":
    main()
