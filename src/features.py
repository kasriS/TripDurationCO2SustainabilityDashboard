# src/features.py
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.cluster import KMeans
R = 6371.0
def haversine_series(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def prepare_data(df, weather_df=None, holiday_df=None, osrm_df=None, fit_kmeans=False, kmeans_pickup=None, kmeans_dropoff=None):
    df = df.copy()
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df['date'] = df['pickup_datetime'].dt.date
    df['hour'] = df['pickup_datetime'].dt.hour.fillna(0).astype(int)
    df['weekday'] = df['pickup_datetime'].dt.weekday.fillna(0).astype(int)
    df['month'] = df['pickup_datetime'].dt.month.fillna(0).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    if weather_df is not None:
        w = weather_df.copy()
        w['date'] = pd.to_datetime(w['date'], dayfirst=True, errors='coerce').dt.date
        w.columns = w.columns.str.strip().str.lower().str.replace(" ", "_")
        if 'precipitation' in w.columns:
            w['precipitation'] = w['precipitation'].replace('T', 0.01).astype(float).fillna(0)
        if 'snow_fall' in w.columns:
            w['snow_fall'] = w['snow_fall'].replace('T', 0.01).astype(float).fillna(0)
        w['precip_intensity'] = w.get('precipitation', 0) + w.get('snow_fall', 0)
        df = df.merge(w[['date','precipitation','snow_fall','precip_intensity']], on='date', how='left')
    if holiday_df is not None:
        hd = holiday_df.copy()
        if 'date' in hd.columns:
            hd['pickup_date'] = pd.to_datetime(hd['date'], errors='coerce').dt.date
        elif 'pickup_date' in hd.columns:
            hd['pickup_date'] = pd.to_datetime(hd['pickup_date'], errors='coerce').dt.date
        else:
            hd['pickup_date'] = pd.Series(dtype='object')
        hd = hd.rename(columns={'holiday':'holiday'}).loc[:, hd.columns.intersection(['pickup_date','holiday'])]
        df['pickup_date'] = df['date']
        df = df.merge(hd, on='pickup_date', how='left')
        df['pickup_holiday'] = df['holiday'].notnull().astype(int)
        df.drop(columns=['holiday','pickup_date'], inplace=True, errors='ignore')
    else:
        df['pickup_holiday'] = 0
    if osrm_df is not None and 'id' in osrm_df.columns:
        df = df.merge(osrm_df[['id','total_distance','total_travel_time','number_of_steps']], on='id', how='left')
    df['osrm_total_distance_km'] = df.get('total_distance', 0) / 1000.0
    df['osrm_total_travel_time_s'] = df.get('total_travel_time', np.nan)
    df['manhattan_dist'] = (df['pickup_latitude'] - df['dropoff_latitude']).abs() + (df['pickup_longitude'] - df['dropoff_longitude']).abs()
    df['haversine_dist'] = haversine_series(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
    df['pickup_geohash'] = df.apply(lambda r: f"{round(r['pickup_latitude']/0.01,2)}_{round(r['pickup_longitude']/0.01,2)}", axis=1)
    freq = df['pickup_geohash'].value_counts().to_dict()
    df['pickup_density'] = df['pickup_geohash'].map(freq).fillna(0)
    lat1 = np.radians(df['pickup_latitude'])
    lat2 = np.radians(df['dropoff_latitude'])
    dlon = np.radians(df['dropoff_longitude'] - df['pickup_longitude'])
    df['bearing'] = np.arctan2(np.sin(dlon) * np.cos(lat2),
                               np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    df['bearing_deg'] = (np.degrees(df['bearing']) + 360) % 360
    df['direction_bin'] = pd.cut(df['bearing_deg'], bins=[0,45,135,225,315,360], labels=['E','N','W','S','E2'])
    df['direction_bin'] = df['direction_bin'].map({'N':0,'E':1,'S':2,'W':3,'E2':1}).fillna(0).astype(int)
    # KMeans not fitted here if fit_kmeans False
    if fit_kmeans and 'pickup_latitude' in df.columns:
        coords_p = df[['pickup_latitude','pickup_longitude']].fillna(0)
        coords_d = df[['dropoff_latitude','dropoff_longitude']].fillna(0)
        kmeans_pickup = KMeans(n_clusters=30, random_state=42).fit(coords_p)
        kmeans_dropoff = KMeans(n_clusters=30, random_state=42).fit(coords_d)
        df['pickup_zone'] = kmeans_pickup.predict(coords_p)
        df['dropoff_zone'] = kmeans_dropoff.predict(coords_d)
    else:
        if 'pickup_zone' not in df.columns:
            df['pickup_zone'] = 0
        if 'dropoff_zone' not in df.columns:
            df['dropoff_zone'] = 0
    try:
        cal = USFederalHolidayCalendar()
        hols = cal.holidays(start=df['pickup_datetime'].min(), end=df['pickup_datetime'].max())
        df['is_holiday'] = df['pickup_datetime'].dt.date.isin(hols.date).astype(int)
    except Exception:
        df['is_holiday'] = 0
    df['rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 16) & (df['hour'] <= 19))).astype(int)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df
