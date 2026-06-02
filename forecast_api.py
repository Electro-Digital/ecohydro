import os
import pickle
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA

# Configuration
ARIMA_MODEL_FILE = "forecasting_dataset/bhepp_inflow_arima.pkl"
ANN_MODEL_FILE = "forecasting_dataset/bhepp_inflow_ann.pkl"
SCALER_FILE = "forecasting_dataset/bhepp_inflow_scaler.pkl"
FORECAST_CSV = "forecasting_dataset/bhepp_7d_forecast_recovery_gen.csv"

HORIZON_HOURS = 7 * 24
RAIN_LAGS = [0, 1, 2, 3, 6, 12, 24]
QO_PER_MW = 0.605
PLANNED_MW = 10.0
DEFAULT_A_EFF = 174897.0
H_MIN = 330.0
H_MAX_TARGET = 332.0
INFLOW_SCALING = 1.0
OM_LAT, OM_LON, OM_TIMEZONE = 14.13, 121.53, "Asia/Manila"

ARIMA_CANDIDATES = [(1,0,0), (2,0,0), (1,0,1), (2,0,1), (2,0,2), (3,0,1), (3,0,2), (1,1,0), (2,1,0), (1,1,1), (2,1,1)]

def fetch_rain_forecast_7d():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": OM_LAT, "longitude": OM_LON, "hourly": "precipitation", "forecast_days": 7, "timezone": OM_TIMEZONE}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    times = pd.to_datetime(data["hourly"]["time"])
    return pd.Series(data["hourly"]["precipitation"], index=times, name="rain_mm")

def build_rain_lag_features(rain_series, lags):
    feat = pd.DataFrame(index=rain_series.index)
    for lag in lags:
        feat[f"rain_lag_{lag}"] = rain_series.shift(lag)
    return feat

def calibrate_A_eff(df):
    Q_out = QO_PER_MW * df["b12_mw"]
    cond = (df["rainfall"] < 0.2) & (Q_out > 0.5) & (df["dam_el_delta"] < -0.005) & df["dam_el_delta"].notna()
    df_dry = df[cond].copy()
    if len(df_dry) < 100:
        return DEFAULT_A_EFF
    A_eff_calc = -Q_out.loc[df_dry.index] * 3600.0 / df_dry["dam_el_delta"]
    A_eff_valid = A_eff_calc[(A_eff_calc > 1e5) & (A_eff_calc < 5e7)].dropna()
    return float(A_eff_valid.median()) if len(A_eff_valid) >= 100 else DEFAULT_A_EFF

def train_model(df):
    """Train ARIMA+ANN model from dataframe with columns: date, b12_mw, dam_el, dam_el_delta, rainfall"""
    df = df.drop_duplicates(subset=['date'], keep='last')
    df = df.set_index("date")
    df = df.asfreq("h")
    df[["b12_mw", "dam_el", "dam_el_delta", "rainfall"]] = df[["b12_mw", "dam_el", "dam_el_delta", "rainfall"]].interpolate(method="time")
    
    A_eff = calibrate_A_eff(df)
    df["Q_out_hist"] = QO_PER_MW * df["b12_mw"]
    df["dam_el_delta_smooth"] = df["dam_el_delta"].rolling(6, center=True, min_periods=1).mean()
    df["Q_in_est"] = (df["Q_out_hist"] + (A_eff * df["dam_el_delta_smooth"]) / 3600.0).rolling(3, center=True, min_periods=1).mean().clip(lower=0.0)
    
    rain_hist = df["rainfall"].rename("rain_mm")
    qin_hist = df["Q_in_est"].rename("Q_in")
    
    X_rain_lags = build_rain_lag_features(rain_hist, RAIN_LAGS)
    df["rain_6h"] = rain_hist.rolling(6).sum()
    df["rain_12h"] = rain_hist.rolling(12).sum()
    df["rain_24h"] = rain_hist.rolling(24).sum()
    df["Q_in_lag1"] = qin_hist.shift(1)
    df["Q_in_lag2"] = qin_hist.shift(2)
    
    X_all = pd.concat([X_rain_lags, df[["rain_6h", "rain_12h", "rain_24h", "Q_in_lag1", "Q_in_lag2"]]], axis=1)
    data = pd.concat([X_all, qin_hist], axis=1).dropna()
    X, y = data.drop(columns=["Q_in"]), data["Q_in"]
    
    if len(data) < 300:
        raise ValueError("Not enough samples to train")
    
    split_idx = int(len(data) * 0.8)
    X_train, X_test, y_train, y_test = X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]
    
    best_arima_model, best_order, best_mae = None, None, np.inf
    for order in ARIMA_CANDIDATES:
        try:
            model = ARIMA(y_train, order=order).fit()
            mae = mean_absolute_error(y_test, model.forecast(steps=len(y_test)))
            if mae < best_mae:
                best_mae, best_arima_model, best_order = mae, model, order
        except:
            continue
    
    if best_arima_model is None:
        raise RuntimeError("All ARIMA candidates failed")
    
    y_train_fitted = best_arima_model.fittedvalues.reindex(y_train.index).bfill().ffill()
    resid_train = (y_train - y_train_fitted).fillna(0.0)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    ann = MLPRegressor(hidden_layer_sizes=(32,16), activation="relu", solver="adam", learning_rate="adaptive", 
                       learning_rate_init=0.001, alpha=0.001, early_stopping=True, validation_fraction=0.15, 
                       n_iter_no_change=20, max_iter=800, random_state=42)
    ann.fit(X_train_s, resid_train)
    
    arima_full = ARIMA(y, order=best_order).fit()
    y_full_fitted = arima_full.fittedvalues.reindex(y.index).bfill().ffill()
    resid_full = (y - y_full_fitted).fillna(0.0)
    
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)
    ann_full = MLPRegressor(hidden_layer_sizes=(32,16), activation="relu", solver="adam", learning_rate="adaptive",
                            learning_rate_init=0.001, alpha=0.001, early_stopping=False, max_iter=800, random_state=42)
    ann_full.fit(X_full_s, resid_full)
    
    os.makedirs(os.path.dirname(ARIMA_MODEL_FILE), exist_ok=True)
    with open(ARIMA_MODEL_FILE, "wb") as f: pickle.dump(arima_full, f)
    with open(ANN_MODEL_FILE, "wb") as f: pickle.dump(ann_full, f)
    with open(SCALER_FILE, "wb") as f: pickle.dump(scaler_full, f)
    
    return {"success": True, "message": f"Model trained with ARIMA{best_order}, MAE: {best_mae:.3f}"}

def run_prediction(df):
    """Run 7-day forecast from dataframe"""
    if not all(os.path.exists(f) for f in [ARIMA_MODEL_FILE, ANN_MODEL_FILE, SCALER_FILE]):
        raise FileNotFoundError("Model files not found. Train model first.")
    
    with open(ARIMA_MODEL_FILE, "rb") as f: arima_model = pickle.load(f)
    with open(ANN_MODEL_FILE, "rb") as f: ann_model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f: scaler = pickle.load(f)
    
    df = df.drop_duplicates(subset=['date'], keep='last')
    df = df.set_index("date")
    df = df.asfreq("h")
    df[["b12_mw", "dam_el", "dam_el_delta", "rainfall"]] = df[["b12_mw", "dam_el", "dam_el_delta", "rainfall"]].interpolate(method="time")
    
    A_eff = calibrate_A_eff(df)
    df["Q_out_hist"] = QO_PER_MW * df["b12_mw"]
    df["dam_el_delta_smooth"] = df["dam_el_delta"].rolling(6, center=True, min_periods=1).mean()
    df["Q_in_est"] = (df["Q_out_hist"] + (A_eff * df["dam_el_delta_smooth"]) / 3600.0).rolling(3, center=True, min_periods=1).mean().clip(lower=0.0)
    
    rain_hist = df["rainfall"].rename("rain_mm")
    qin_hist = df["Q_in_est"].rename("Q_in")
    
    X_rain_lags = build_rain_lag_features(rain_hist, RAIN_LAGS)
    df["rain_6h"] = rain_hist.rolling(6).sum()
    df["rain_12h"] = rain_hist.rolling(12).sum()
    df["rain_24h"] = rain_hist.rolling(24).sum()
    df["Q_in_lag1"] = qin_hist.shift(1)
    df["Q_in_lag2"] = qin_hist.shift(2)
    
    X_all = pd.concat([X_rain_lags, df[["rain_6h", "rain_12h", "rain_24h", "Q_in_lag1", "Q_in_lag2"]]], axis=1)
    data = pd.concat([X_all, qin_hist], axis=1).dropna()
    X = data.drop(columns=["Q_in"])
    
    last_ts, last_H = df.index[-1], df["dam_el"].iloc[-1]
    rain_fore = fetch_rain_forecast_7d()
    forecast_index = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=HORIZON_HOURS, freq="h")
    combined_rain = pd.concat([rain_hist, rain_fore])[~pd.concat([rain_hist, rain_fore]).index.duplicated(keep="last")]
    
    arima_fore = pd.Series(arima_model.get_forecast(steps=HORIZON_HOURS).predicted_mean.values, index=forecast_index)
    qin_combined = qin_hist.copy()
    q_in_fore_values = []
    
    for t in forecast_index:
        q_arima = arima_fore.loc[t]
        row = {f"rain_lag_{lag}": combined_rain.get(t - pd.Timedelta(hours=lag), 0.0) for lag in RAIN_LAGS}
        row.update({"rain_6h": combined_rain.loc[:t].tail(6).sum(), "rain_12h": combined_rain.loc[:t].tail(12).sum(), 
                    "rain_24h": combined_rain.loc[:t].tail(24).sum()})
        row["Q_in_lag1"] = qin_combined.iloc[-1] if len(qin_combined) >= 1 else 0.0
        row["Q_in_lag2"] = qin_combined.iloc[-2] if len(qin_combined) >= 2 else row["Q_in_lag1"]
        row_df = pd.DataFrame([row], index=[t])[X.columns]
        q_hybrid = max((q_arima + ann_model.predict(scaler.transform(row_df))[0]) * INFLOW_SCALING, 0.0)
        q_in_fore_values.append(q_hybrid)
        qin_combined.loc[t] = q_hybrid
    
    Q_in_fore_smooth = pd.Series(q_in_fore_values, index=forecast_index).rolling(3, center=True, min_periods=1).mean()
    
    H_vals, Q_out_vals, mode_vals = [], [], []
    H, mode = last_H, "recovery"
    
    for t in forecast_index:
        q_in = max(Q_in_fore_smooth.loc[t], 0.0)
        if mode == "recovery":
            q_out = 0.0
            H_next = H + (q_in - q_out) * 3600.0 / A_eff
            if H_next >= H_MAX_TARGET:
                H_next, mode = H_MAX_TARGET, "generation"
        else:
            q_out = QO_PER_MW * PLANNED_MW
            H_next = H + (q_in - q_out) * 3600.0 / A_eff
            if H_next <= H_MIN:
                H_next, mode, q_out = H_MIN, "recovery", 0.0
        H_vals.append(H_next)
        Q_out_vals.append(q_out)
        mode_vals.append(mode)
        H = H_next
    
    result = pd.DataFrame({
        "rain_mm": rain_fore.reindex(forecast_index).fillna(0.0),
        "Q_in_cms": Q_in_fore_smooth,
        "Q_out_cms": Q_out_vals,
        "dam_el_masl": H_vals,
        "MW_schedule": [0.0 if m == "recovery" else PLANNED_MW for m in mode_vals],
        "operation_mode": mode_vals
    })
    
    result.to_csv(FORECAST_CSV)
    return {"success": True, "message": "Forecast completed", "csv_path": FORECAST_CSV}

def get_forecast_data():
    """Read forecast CSV and return as JSON"""
    if not os.path.exists(FORECAST_CSV):
        return {"error": "No forecast data available. Run prediction first."}
    
    df = pd.read_csv(FORECAST_CSV, index_col=0)
    return {
        "timestamps": df.index.tolist(),
        "rain_mm": df["rain_mm"].tolist(),
        "Q_in_cms": df["Q_in_cms"].tolist(),
        "Q_out_cms": df["Q_out_cms"].tolist(),
        "dam_el_masl": df["dam_el_masl"].tolist(),
        "MW_schedule": df["MW_schedule"].tolist(),
        "operation_mode": df["operation_mode"].tolist()
    }
