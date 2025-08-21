# src/predict.py
# Enhanced prediction functions with complete ML workflow

import os
import math
import joblib
import numpy as np
import pandas as pd
import logging

# Paths to model artifacts
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "taxi_duration_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "selected_features.pkl")

# Constants for workflow calculations
AVERAGE_SPEED_KMH = 25.0  # NYC taxi average speed
FUEL_EFFICIENCY_L_PER_100KM = 9.5  # Liters per 100km
CO2_PER_LITER_FUEL = 2.31  # kg CO2 per liter gasoline
DIRECT_EMISSION_FACTOR = 0.155  # kg CO2 per km (alternative method)

# Lazy-loaded artifacts
_model = None
_scaler = None
_selected_features = None
_model_err = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_artifacts():
    """Lazy-load model artifacts. Keep going even if missing."""
    global _model, _scaler, _selected_features, _model_err
    if _model is not None or _model_err is not None:
        return

    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
            _model = joblib.load(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
            _selected_features = joblib.load(FEATURES_PATH)
            logger.info("✅ Model artifacts loaded successfully")
        else:
            _model_err = "Model files not found in /models. Using fallback estimations."
            logger.warning("⚠️  Model files not found - using fallback predictions")
    except Exception as e:
        _model_err = f"Could not load model artifacts: {e}"
        logger.error(f"❌ Error loading models: {e}")


def predict_duration(df_prep):
    """
    STEP 1: Predict trip duration using ML model
    Returns duration in seconds
    """
    global _model, _scaler, _selected_features, _model_err

    if _model_err:
        logger.warning(f"Using fallback estimation: {_model_err}")
        return None

    if _model is None or _scaler is None or _selected_features is None:
        logger.warning("Model artifacts not loaded")
        return None

    try:
        # Ensure we have all selected features
        for f in _selected_features:
            if f not in df_prep.columns:
                df_prep[f] = 0

        X = df_prep[_selected_features].fillna(0)
        Xs = _scaler.transform(X)
        pred_log = _model.predict(Xs)[0]
        duration_seconds = float(np.expm1(pred_log))

        logger.info(f"🤖 ML Prediction: {duration_seconds / 60:.1f} minutes")
        return duration_seconds

    except Exception as e:
        logger.error(f"Error in ML prediction: {e}")
        return None


def estimate_distance_from_duration(duration_seconds, avg_speed_kmh=AVERAGE_SPEED_KMH):
    """
    STEP 2: Derive distance from predicted duration and average speed
    """
    if duration_seconds is None or duration_seconds <= 0:
        return None

    duration_hours = duration_seconds / 3600
    distance_km = duration_hours * avg_speed_kmh

    logger.info(f"📏 Estimated distance: {distance_km:.2f} km from {duration_seconds / 60:.1f} min trip")
    return distance_km


def calculate_fuel_consumption(distance_km):
    """
    STEP 3: Calculate fuel consumption from distance
    Returns fuel in liters
    """
    if distance_km is None or distance_km <= 0:
        return 0

    fuel_liters = (distance_km * FUEL_EFFICIENCY_L_PER_100KM) / 100
    logger.info(f"⛽ Fuel consumption: {fuel_liters:.3f} L for {distance_km:.2f} km")
    return fuel_liters


def calculate_co2_from_fuel(fuel_liters):
    """
    STEP 4: Calculate CO2 emissions from fuel consumption
    Returns CO2 in kg
    """
    if fuel_liters is None or fuel_liters <= 0:
        return 0

    co2_kg = fuel_liters * CO2_PER_LITER_FUEL
    logger.info(f"🌍 CO2 emissions: {co2_kg:.3f} kg from {fuel_liters:.3f} L fuel")
    return co2_kg


def estimate_co2_direct(distance_km):
    """
    Alternative STEP 4: Direct CO2 calculation from distance
    Returns CO2 in kg using direct emission factor
    """
    if distance_km is None or distance_km <= 0:
        return 0

    co2_kg = distance_km * DIRECT_EMISSION_FACTOR
    logger.info(f"🌍 CO2 (direct): {co2_kg:.3f} kg from {distance_km:.2f} km")
    return co2_kg


def estimate_co2_saving(orig_distance_km, optimized_distance_km):
    """
    Calculate CO₂ savings from route optimization
    """
    if isinstance(orig_distance_km, (float, int)) and isinstance(optimized_distance_km, (float, int)):
        distance_saved = orig_distance_km - optimized_distance_km
        co2_saved = distance_saved * DIRECT_EMISSION_FACTOR
        logger.info(f"✨ CO2 saved: {co2_saved:.3f} kg from {distance_saved:.2f} km reduction")
        return co2_saved
    else:
        # Handle array inputs
        distance_saved = np.array(orig_distance_km) - np.array(optimized_distance_km)
        co2_saved = distance_saved * DIRECT_EMISSION_FACTOR
        return co2_saved


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance between two points in kilometers"""
    R = 6371.0  # Earth radius in km

    # Validate coordinates
    if not all(isinstance(coord, (int, float)) for coord in [lat1, lon1, lat2, lon2]):
        logger.error("Invalid coordinates provided to haversine function")
        return None

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(dlon / 2.0) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c

    logger.info(f"📍 Haversine distance: {distance:.2f} km")
    return distance


def complete_ml_workflow(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                         pickup_datetime, passenger_count=1, avg_speed_kmh=AVERAGE_SPEED_KMH,
                         optimization_pct=8):
    """
    Complete ML workflow implementation:
    Step 1: ML prediction → Step 2: Distance → Step 3: Fuel → Step 4: CO₂
    """
    logger.info("🚀 Starting complete ML workflow...")

    # Validate input coordinates
    if not all(isinstance(coord, (int, float)) for coord in [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]):
        logger.error("Invalid coordinates provided to workflow")
        return None

    # Prepare input data
    df = pd.DataFrame([{
        "id": "workflow_complete",
        "pickup_datetime": pickup_datetime,
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,
        "passenger_count": passenger_count
    }])

    results = {}

    try:
        # Import features module for data preparation
        from features import prepare_data
        df_prep = prepare_data(df, fit_kmeans=False)

        # Calculate straight-line distance first
        straight_line_km = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        if straight_line_km is None or straight_line_km <= 0:
            logger.error("Invalid straight-line distance calculation")
            return None

        # STEP 1: ML Duration Prediction
        duration_s = predict_duration(df_prep)
        if duration_s is None:
            # Fallback calculation with realistic traffic factor (1.4x for NYC)
            traffic_factor = 1.4
            estimated_route_km = straight_line_km * traffic_factor
            duration_s = (estimated_route_km / avg_speed_kmh) * 3600
            logger.info(f"🔄 Fallback duration: {duration_s / 60:.1f} min")

        results['predicted_duration_s'] = duration_s
        results['predicted_duration_min'] = duration_s / 60

        # STEP 2: Distance Derivation
        estimated_route_km = estimate_distance_from_duration(duration_s, avg_speed_kmh)

        # Validate route distance is realistic (at least straight line distance)
        if estimated_route_km is None or estimated_route_km < straight_line_km:
            # Use realistic traffic factor if route distance is invalid
            traffic_factor = 1.4
            estimated_route_km = straight_line_km * traffic_factor
            logger.warning("🔄 Using traffic factor for route distance estimation")

        results['straight_line_km'] = straight_line_km
        results['estimated_route_km'] = estimated_route_km

        # Calculate route efficiency (should be <= 100%)
        if estimated_route_km > 0:
            route_efficiency = min(100, (straight_line_km / estimated_route_km) * 100)
        else:
            route_efficiency = 0
        results['route_efficiency'] = route_efficiency

        # STEP 3: Fuel Consumption
        fuel_consumed = calculate_fuel_consumption(estimated_route_km)
        optimized_route_km = estimated_route_km * (100 - optimization_pct) / 100
        optimized_fuel = calculate_fuel_consumption(optimized_route_km)
        fuel_saved = fuel_consumed - optimized_fuel

        results['original_fuel_L'] = fuel_consumed
        results['optimized_fuel_L'] = optimized_fuel
        results['fuel_saved_L'] = fuel_saved
        results['optimized_route_km'] = optimized_route_km

        # STEP 4: CO₂ Emissions
        original_co2 = calculate_co2_from_fuel(fuel_consumed)
        optimized_co2 = calculate_co2_from_fuel(optimized_fuel)
        co2_saved = original_co2 - optimized_co2

        results['original_co2_kg'] = original_co2
        results['optimized_co2_kg'] = optimized_co2
        results['co2_saved_kg'] = co2_saved
        results['optimization_pct'] = optimization_pct

        # Additional metrics
        results['trees_equivalent'] = co2_saved / 0.06  # ~60g CO2 per tree per day
        results['cost_savings_usd'] = fuel_saved * 1.2  # ~$1.2 per liter

        logger.info("✅ Complete ML workflow finished successfully!")
        return results

    except Exception as e:
        logger.error(f"❌ Error in ML workflow: {e}")
        return None


def batch_workflow_analysis(trips_df, optimization_levels=[6, 12, 18]):
    """
    Run workflow analysis on batch of trips for different optimization levels
    """
    logger.info(f"📊 Running batch analysis on {len(trips_df)} trips...")

    results_summary = []

    for opt_level in optimization_levels:
        total_co2_saved = 0
        total_fuel_saved = 0
        total_cost_saved = 0
        valid_trips = 0

        for idx, trip in trips_df.iterrows():
            try:
                workflow_result = complete_ml_workflow(
                    trip['pickup_latitude'], trip['pickup_longitude'],
                    trip['dropoff_latitude'], trip['dropoff_longitude'],
                    trip.get('pickup_datetime', '2016-01-01 12:00:00'),
                    trip.get('passenger_count', 1),
                    AVERAGE_SPEED_KMH,
                    opt_level
                )

                if workflow_result:
                    total_co2_saved += workflow_result['co2_saved_kg']
                    total_fuel_saved += workflow_result['fuel_saved_L']
                    total_cost_saved += workflow_result['cost_savings_usd']
                    valid_trips += 1

            except Exception as e:
                logger.warning(f"⚠️  Error processing trip {idx}: {e}")
                continue

        results_summary.append({
            'optimization_level': opt_level,
            'valid_trips': valid_trips,
            'total_co2_saved_kg': total_co2_saved,
            'total_fuel_saved_L': total_fuel_saved,
            'total_cost_saved_usd': total_cost_saved,
            'avg_co2_saved_per_trip': total_co2_saved / valid_trips if valid_trips > 0 else 0,
            'trees_equivalent': total_co2_saved / 0.06,
        })

        logger.info(f"📈 Optimization {opt_level}%: {total_co2_saved:.1f} kg CO₂ saved across {valid_trips} trips")

    return pd.DataFrame(results_summary)


def get_workflow_summary():
    """Return summary information about the ML workflow"""
    return {
        'workflow_steps': [
            "Step 1: ML Model predicts trip duration from pickup/dropoff/time features",
            "Step 2: Derive distance traveled using predicted duration and average speed",
            "Step 3: Calculate fuel consumption based on distance and efficiency",
            "Step 4: Convert fuel consumption to CO₂ emissions and show optimization savings"
        ],
        'constants': {
            'average_speed_kmh': AVERAGE_SPEED_KMH,
            'fuel_efficiency_L_per_100km': FUEL_EFFICIENCY_L_PER_100KM,
            'co2_per_liter_fuel': CO2_PER_LITER_FUEL,
            'direct_emission_factor': DIRECT_EMISSION_FACTOR
        },
        'model_info': {
            'model_path': MODEL_PATH,
            'scaler_path': SCALER_PATH,
            'features_path': FEATURES_PATH,
            'model_loaded': _model is not None,
            'model_error': _model_err
        }
    }
