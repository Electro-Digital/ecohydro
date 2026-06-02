import os
import sys
import pickle
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymysql

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB

# ======================================================
# CONFIGURATION
# ======================================================

# ---- MODE ----
TRAIN_MODE = False          # Training mode enabled
DO_PLOT    = True         # Disable plots for API usage

# Use hybrid (ARIMA + ANN) for forecast?
# You can set this to False later if you want pure ARIMA-only in forecast mode.
USE_HYBRID_FOR_FORECAST = True

# ---- FILES ----
ARIMA_MODEL_FILE  = "bhepp_inflow_arima.pkl"
ANN_MODEL_FILE    = "bhepp_inflow_ann.pkl"
SCALER_FILE       = "bhepp_inflow_scaler.pkl"
FORECAST_CSV      = "bhepp_7d_forecast_recovery_gen.csv"

# ---- FORECAST HORIZON ----
HORIZON_HOURS = 7 * 24  # 7 days

# ---- RAINFALL LAGS (hrs) ----
RAIN_LAGS = [0, 1, 2, 3, 6, 12, 24]

# ---- PLANT / RESERVOIR PARAMETERS ----
QO_PER_MW       = 0.605          # m³/s per MW (10 MW ≈ 6 m³/s)
PLANNED_MW      = 10.0           # MW during generation phase
DEFAULT_A_EFF   = 174897.0        # m², used if calibration fails
H_MIN           = 330.0          # masl, minimum operating level
H_MAX_TARGET    = 332.0          # masl, target max level (start generation once reached)
INFLOW_SCALING  = 1.0            # scenario scaling (1.0 normal, <1 dry, >1 wet)

# ---- ARIMA CANDIDATES ----
# We'll search over these (p,d,q) orders and pick the best on validation MAE.
ARIMA_CANDIDATES = [
    (1, 0, 0),
    (2, 0, 0),
    (1, 0, 1),
    (2, 0, 1),
    (2, 0, 2),
    (3, 0, 1),
    (3, 0, 2),
    (1, 1, 0),
    (2, 1, 0),
    (1, 1, 1),
    (2, 1, 1),
]

# ---- OPEN-METEO COORDINATES ----
OM_LAT      = 14.13
OM_LON      = 121.53
OM_TIMEZONE = "Asia/Manila"


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def fetch_rain_forecast_7d():
    """Fetch 7-day hourly precipitation (mm) from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": OM_LAT,
        "longitude": OM_LON,
        "hourly": "precipitation",
        "forecast_days": 7,
        "timezone": OM_TIMEZONE,
    }
    print(f"\nRequesting 7-day rainfall forecast from Open-Meteo for "
          f"lat={OM_LAT}, lon={OM_LON} ...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data or "time" not in data["hourly"] or "precipitation" not in data["hourly"]:
        raise ValueError("Open-Meteo response missing 'hourly/precipitation' data.")

    times = pd.to_datetime(data["hourly"]["time"])
    prec  = pd.Series(data["hourly"]["precipitation"], index=times, name="rain_mm")
    return prec


def build_rain_lag_features(rain_series: pd.Series, lags):
    """Create a DataFrame with columns rain_lag_0, rain_lag_1, ... for given lags (in hours)."""
    feat = pd.DataFrame(index=rain_series.index)
    for lag in lags:
        feat[f"rain_lag_{lag}"] = rain_series.shift(lag)
    return feat


def calibrate_A_eff_from_dry_periods(df, min_samples=100, rain_thresh=0.2,
                                     min_Qo=0.5, min_dH=0.005):
    """
    Estimate effective surface area A_eff from dry periods, assuming inflow ~ 0.
    Uses:
        A_eff ≈ - Q_out * 3600 / dH  (when dH < 0, Qo > 0, rainfall small)
    """
    if "b12_Qo" in df.columns and not df["b12_Qo"].isna().all():
        Q_out = df["b12_Qo"]
    else:
        Q_out = QO_PER_MW * df["b12_mw"]

    cond = (
        (df["rainfall"] < rain_thresh) &
        (Q_out > min_Qo) &
        (df["dam_el_delta"] < -min_dH) &
        df["dam_el_delta"].notna()
    )

    df_dry = df[cond].copy()
    if len(df_dry) < min_samples:
        print(f"\n[WARN] Not enough dry-period samples for A_eff calibration "
              f"({len(df_dry)} found). Using DEFAULT_A_EFF = {DEFAULT_A_EFF:.0f} m².")
        return DEFAULT_A_EFF

    A_eff_calc = -Q_out.loc[df_dry.index] * 3600.0 / df_dry["dam_el_delta"]

    # Filter out unrealistic values
    A_eff_valid = A_eff_calc[(A_eff_calc > 1e5) & (A_eff_calc < 5e7)].dropna()

    if len(A_eff_valid) < min_samples:
        print(f"\n[WARN] Not enough valid A_eff samples ({len(A_eff_valid)}). "
              f"Using DEFAULT_A_EFF = {DEFAULT_A_EFF:.0f} m².")
        return DEFAULT_A_EFF

    A_eff_median = A_eff_valid.median()
    print(f"\nCalibrated A_eff from data: {A_eff_median:.0f} m² "
          f"(median of {len(A_eff_valid)} samples)")

    return float(A_eff_median)


def fetch_data_from_db():
    """Fetch data from b01_parameters_hourly and b02_parameters_hourly tables."""
    print("\n=== FETCHING DATA FROM DATABASE ===")
    
    conn = pymysql.connect(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DB
    )
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    cursor.execute("SELECT ts, mw, upper_water, rr FROM b01_parameters_hourly ORDER BY ts")
    b01_data = cursor.fetchall()
    
    cursor.execute("SELECT ts, mw, upper_water, rr FROM b02_parameters_hourly ORDER BY ts")
    b02_data = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    df_b01 = pd.DataFrame(b01_data)
    df_b02 = pd.DataFrame(b02_data)
    
    df = pd.merge(df_b01, df_b02, on='ts', suffixes=('_b01', '_b02'), how='outer')
    df = df.sort_values('ts').reset_index(drop=True)
    
    df['date'] = pd.to_datetime(df['ts'])
    df['b01_mw'] = df['mw_b01']
    df['b02_mw'] = df['mw_b02']
    df['b12_mw'] = df['b01_mw'] + df['b02_mw']
    df['dam_el'] = df['upper_water_b01']
    df['rainfall'] = df[['rr_b01', 'rr_b02']].mean(axis=1)
    df['dam_el_delta'] = df['dam_el'].diff()
    
    df = df[['date', 'b01_mw', 'b02_mw', 'b12_mw', 'dam_el', 'dam_el_delta', 'rainfall']]
    
    print(f"Fetched {len(df)} records from database")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def evaluate_series(y_true, y_pred, label="model"):
    """Convenience for MAE, RMSE, R² printing."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2   = r2_score(y_true, y_pred)
    print(f"\n=== {label} ===")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")
    return mae, rmse, r2


# ======================================================
# 1. LOAD & CLEAN DATA FROM DATABASE
# ======================================================

print("\n=== LOADING DATA FROM DATABASE ===")
df = fetch_data_from_db()
df = df.drop_duplicates(subset=['date'], keep='last')
df = df.set_index("date")

numeric_cols = ["b12_mw", "dam_el", "dam_el_delta", "rainfall"]
if "b12_Qo" in df.columns:
    numeric_cols.append("b12_Qo")

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.asfreq("h")
df[numeric_cols] = df[numeric_cols].interpolate(method="time")

print("\nColumn dtypes (after cleaning):")
print(df.dtypes[numeric_cols])

has_measured_Qo = "b12_Qo" in df.columns and not df["b12_Qo"].isna().all()

if has_measured_Qo:
    df["Q_out_hist"] = df["b12_Qo"]
    print("\nUsing measured b12_Qo as historical Q_out.")
else:
    df["Q_out_hist"] = QO_PER_MW * df["b12_mw"]
    print("\nUsing Q_out_hist = 0.6 * b12_mw.")


# ======================================================
# 2. CALIBRATE A_eff AND COMPUTE Q_in_est WITH SMOOTHING
# ======================================================

A_eff = calibrate_A_eff_from_dry_periods(df)
print(f"Effective surface area A_eff = {A_eff:.0f} m² (used for water balance).")

df["dam_el_delta_smooth"] = df["dam_el_delta"].rolling(6, center=True, min_periods=1).mean()

df["Q_in_est_raw"] = df["Q_out_hist"] + (A_eff * df["dam_el_delta_smooth"]) / 3600.0
df["Q_in_est"] = df["Q_in_est_raw"].rolling(3, center=True, min_periods=1).mean()
df["Q_in_est"] = df["Q_in_est"].clip(lower=0.0)

rain_hist = df["rainfall"].rename("rain_mm")
qin_hist  = df["Q_in_est"].rename("Q_in")


# ======================================================
# 3. FEATURE ENGINEERING FOR INFLOW MODEL
# ======================================================

# Rainfall lag features
X_rain_lags = build_rain_lag_features(rain_hist, RAIN_LAGS)

# Rolling rainfall sums
df["rain_6h"]  = rain_hist.rolling(6).sum()
df["rain_12h"] = rain_hist.rolling(12).sum()
df["rain_24h"] = rain_hist.rolling(24).sum()

# Q_in lags
df["Q_in_lag1"] = qin_hist.shift(1)
df["Q_in_lag2"] = qin_hist.shift(2)

# Final feature set for ANN
X_all = pd.concat(
    [
        X_rain_lags,
        df[["rain_6h", "rain_12h", "rain_24h", "Q_in_lag1", "Q_in_lag2"]],
    ],
    axis=1
)

data = pd.concat([X_all, qin_hist], axis=1).dropna()
X = data.drop(columns=["Q_in"])
y = data["Q_in"]

print(f"\nTotal samples after lagging & features: {len(data)}")


# ======================================================
# 4. TRAINING MODE (ARIMA ORDER SEARCH + HYBRID ANN)
# ======================================================

if TRAIN_MODE:
    print("\n=== TRAINING MODE: HYBRID ARIMA + ANN WITH ARIMA ORDER SEARCH ===")

    if len(data) < 300:
        raise ValueError("Not enough samples to train (need at least ~300).")

    # 80/20 chronological split
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)}")
    print(f"Test size : {len(X_test)}")

    # ---- 4.1 Baseline model (mean inflow) ----
    y_base = np.full_like(y_test.values, y_train.mean())
    mae_base, rmse_base, r2_base = evaluate_series(y_test, y_base, label="BASELINE (Mean Q_in)")

    # ---- 4.2 ARIMA order search on training Q_in ----
    best_arima_model = None
    best_order       = None
    best_mae_arima   = np.inf
    best_metrics_arima = None

    print("\nSearching best ARIMA order from candidates:")
    for order in ARIMA_CANDIDATES:
        try:
            print(f"  Trying ARIMA{order} ...", end="")
            model = ARIMA(y_train, order=order).fit()
            # Forecast on test portion
            y_arima_test = model.forecast(steps=len(y_test))
            mae_a = mean_absolute_error(y_test, y_arima_test)
            print(f" MAE={mae_a:.3f}")

            if mae_a < best_mae_arima:
                best_mae_arima = mae_a
                best_arima_model = model
                best_order = order
                rmse_a = mean_squared_error(y_test, y_arima_test) ** 0.5
                r2_a   = r2_score(y_test, y_arima_test)
                best_metrics_arima = (mae_a, rmse_a, r2_a)
        except Exception as e:
            print(f" failed ({e})")
            continue

    if best_arima_model is None:
        raise RuntimeError("All ARIMA candidates failed to fit. Please adjust ARIMA_CANDIDATES.")

    print(f"\nBest ARIMA order selected: {best_order}")
    mae_arima, rmse_arima, r2_arima = best_metrics_arima
    print("\n=== ARIMA-ONLY (best order) on Test Set ===")
    print(f"MAE  : {mae_arima:.3f} m³/s")
    print(f"RMSE : {rmse_arima:.3f} m³/s")
    print(f"R²   : {r2_arima:.3f}")

    # ---- 4.3 ANN on ARIMA residuals ----
    # Residuals on training data
    y_train_fitted = best_arima_model.fittedvalues
    y_train_fitted = y_train_fitted.reindex(y_train.index).bfill().ffill()
    resid_train = (y_train - y_train_fitted).fillna(0.0)

    scaler_tmp = StandardScaler()
    X_train_s = scaler_tmp.fit_transform(X_train)

    ann_tmp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        alpha=0.001,           # L2 regularization
        early_stopping=True,   # uses internal validation
        validation_fraction=0.15,
        n_iter_no_change=20,
        max_iter=800,
        random_state=42
    )

    print("\nTraining ANN on ARIMA residuals...")
    ann_tmp.fit(X_train_s, resid_train)

    # ---- 4.4 Evaluate Hybrid ARIMA+ANN on test set ----
    X_test_s = scaler_tmp.transform(X_test)

    # ARIMA-only forecast on test
    y_arima_test = best_arima_model.forecast(steps=len(y_test))
    y_arima_test = pd.Series(y_arima_test, index=y_test.index)

    # ANN residual correction
    resid_test_pred = ann_tmp.predict(X_test_s)

    # Hybrid prediction
    y_hybrid_test = (y_arima_test + resid_test_pred)
    y_hybrid_test = y_hybrid_test.clip(lower=0.0)

    mae_h, rmse_h, r2_h = evaluate_series(y_test, y_hybrid_test, label="HYBRID (ARIMA + ANN)")

    # Compare models
    print("\n=== SUMMARY COMPARISON (on test set) ===")
    print(f"Baseline MAE : {mae_base:.3f}, R² : {r2_base:.3f}")
    print(f"ARIMA   MAE : {mae_arima:.3f}, R² : {r2_arima:.3f}")
    print(f"Hybrid  MAE : {mae_h:.3f}, R² : {r2_h:.3f}")

    # Optional rule-of-thumb: warn if hybrid is worse than ARIMA
    if mae_h > mae_arima:
        print("\n[WARN] Hybrid model MAE is worse than ARIMA-only on test set.")
        print("       You may set USE_HYBRID_FOR_FORECAST = False if this persists.")

    # ------------------------------------------------------
    # 4.5 Train FINAL models on FULL data for deployment
    # ------------------------------------------------------
    print("\nFitting FINAL ARIMA on full Q_in series with best order...")
    arima_full = ARIMA(y, order=best_order).fit()

    y_full_fitted = arima_full.fittedvalues
    y_full_fitted = y_full_fitted.reindex(y.index).bfill().ffill()
    resid_full = (y - y_full_fitted).fillna(0.0)

    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)

    print("Training FINAL ANN on full residual series...")
    ann_full = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        alpha=0.001,
        early_stopping=False,  # use all data
        max_iter=800,
        random_state=42
    )
    ann_full.fit(X_full_s, resid_full)

    # ---- Save models and scaler ----
    with open(ARIMA_MODEL_FILE, "wb") as f:
        pickle.dump(arima_full, f)
    with open(ANN_MODEL_FILE, "wb") as f:
        pickle.dump(ann_full, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler_full, f)

    print(f"\nSaved ARIMA model to : {ARIMA_MODEL_FILE}")
    print(f"Saved ANN model to   : {ANN_MODEL_FILE}")
    print(f"Saved scaler to      : {SCALER_FILE}")

    # ---- Backtest dam_el over last 7 days using hybrid inflow ----
    BACKTEST_DAYS  = 7
    BACKTEST_HOURS = BACKTEST_DAYS * 24

    if len(df) > BACKTEST_HOURS + 200:
        print(f"\n=== BACKTEST dam_el over last {BACKTEST_DAYS} days (Hybrid ARIMA+ANN) ===")

        df_back = df.iloc[-BACKTEST_HOURS:].copy()
        H_true  = df_back["dam_el"]
        mw_bt   = df_back["b12_mw"]

        full_feat = pd.concat(
            [
                build_rain_lag_features(rain_hist, RAIN_LAGS),
                df[["rain_6h", "rain_12h", "rain_24h", "Q_in_lag1", "Q_in_lag2"]],
            ],
            axis=1
        ).dropna()

        common_idx = df_back.index.intersection(full_feat.index)
        X_bt = full_feat.loc[common_idx]
        H_true_bt = H_true.loc[common_idx]
        mw_bt     = mw_bt.loc[common_idx]

        # Compute inflow using hybrid model for backtest window
        X_bt_s = scaler_full.transform(X_bt[X.columns])

        # ARIMA component
        arima_bt_pred = arima_full.get_prediction(start=common_idx[0], end=common_idx[-1])
        q_in_arima_bt = arima_bt_pred.predicted_mean
        q_in_arima_bt = q_in_arima_bt.reindex(common_idx)

        # ANN residuals
        resid_bt_pred = ann_full.predict(X_bt_s)

        Q_in_bt = (q_in_arima_bt + resid_bt_pred)
        Q_in_bt = pd.Series(Q_in_bt.clip(lower=0.0), index=common_idx, name="Q_in_bt")

        Q_in_bt_smooth = Q_in_bt.rolling(3, center=True, min_periods=1).mean()

        H_sim_vals = []
        H = H_true_bt.iloc[0]

        for t in common_idx:
            q_in  = max(Q_in_bt_smooth.loc[t], 0.0)
            q_out = QO_PER_MW * mw_bt.loc[t]
            H_next = H + (q_in - q_out) * 3600.0 / A_eff
            if H_next < H_MIN:
                H_next = H_MIN
            H_sim_vals.append(H_next)
            H = H_next

        H_sim = pd.Series(H_sim_vals, index=common_idx, name="H_sim")

        mae_H  = mean_absolute_error(H_true_bt, H_sim)
        mse_H  = mean_squared_error(H_true_bt, H_sim)
        rmse_H = mse_H ** 0.5

        print(f"dam_el MAE  : {mae_H:.3f} m")
        print(f"dam_el RMSE : {rmse_H:.3f} m")

        if DO_PLOT:
            plt.figure(figsize=(10, 5))
            H_true_bt.plot(label="Actual dam_el")
            H_sim.plot(label="Simulated dam_el (Hybrid)", linestyle="--")
            plt.axhline(H_MIN, color="red", linestyle=":", label=f"H_min={H_MIN}")
            plt.title(f"Backtest: Dam Elevation (last {BACKTEST_DAYS} days, Hybrid ARIMA+ANN)")
            plt.xlabel("Date")
            plt.ylabel("Elevation (masl)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    print("\nTraining mode finished.")
    raise SystemExit()


# ======================================================
# 5. FORECAST MODE (CYCLIC RECOVERY ↔ GENERATION, HYBRID)
# ======================================================

print("\n=== FORECAST MODE: HYBRID ARIMA + ANN ===")

if not os.path.exists(ARIMA_MODEL_FILE) or not os.path.exists(ANN_MODEL_FILE) or not os.path.exists(SCALER_FILE):
    raise FileNotFoundError(
        "Model files not found. Run once with TRAIN_MODE = True first."
    )

with open(ARIMA_MODEL_FILE, "rb") as f:
    arima_model = pickle.load(f)
with open(ANN_MODEL_FILE, "rb") as f:
    ann_model = pickle.load(f)
with open(SCALER_FILE, "rb") as f:
    scaler = pickle.load(f)

last_ts = df.index[-1]
last_H  = df["dam_el"].iloc[-1]
print(f"\nLast historical timestamp: {last_ts}")
print(f"Last historical dam_el   : {last_H:.3f} masl")

# --- 5.1 Rainfall forecast for 7 days ---
rain_fore = fetch_rain_forecast_7d()

forecast_index = pd.date_range(
    start=last_ts + pd.Timedelta(hours=1),
    periods=HORIZON_HOURS,
    freq="h"
)

combined_rain = pd.concat([rain_hist, rain_fore])
combined_rain = combined_rain[~combined_rain.index.duplicated(keep="last")]

# --- 5.2 ARIMA forecast (base inflow) ---
print(f"\nComputing ARIMA forecast for Q_in...")
arima_fore_res = arima_model.get_forecast(steps=HORIZON_HOURS)
arima_fore_mean = arima_fore_res.predicted_mean
arima_fore = pd.Series(arima_fore_mean.values, index=forecast_index, name="Q_in_arima_fore")

# --- 5.3 Hybrid inflow forecast (sequential, updating Q_in lags) ---
qin_combined = qin_hist.copy()   # historical + forecasted Q_in
q_in_fore_values = []

for t in forecast_index:
    # Base ARIMA prediction at time t
    q_arima = arima_fore.loc[t]

    if USE_HYBRID_FOR_FORECAST:
        row = {}

        # lagged rainfall
        for lag in RAIN_LAGS:
            t_lag = t - pd.Timedelta(hours=lag)
            row[f"rain_lag_{lag}"] = combined_rain.get(t_lag, 0.0)

        # rolling rainfall sums up to current t
        window_6  = combined_rain.loc[:t].tail(6).sum()
        window_12 = combined_rain.loc[:t].tail(12).sum()
        window_24 = combined_rain.loc[:t].tail(24).sum()
        row["rain_6h"]  = window_6
        row["rain_12h"] = window_12
        row["rain_24h"] = window_24

        # Q_in lags (use combined history including previous forecasts)
        if len(qin_combined) >= 2:
            row["Q_in_lag1"] = qin_combined.iloc[-1]
            row["Q_in_lag2"] = qin_combined.iloc[-2]
        elif len(qin_combined) == 1:
            row["Q_in_lag1"] = qin_combined.iloc[-1]
            row["Q_in_lag2"] = qin_combined.iloc[-1]
        else:
            row["Q_in_lag1"] = 0.0
            row["Q_in_lag2"] = 0.0

        row_df = pd.DataFrame([row], index=[t])
        row_df = row_df[X.columns]  # ensure same order/columns as training

        row_s = scaler.transform(row_df)
        resid_corr = ann_model.predict(row_s)[0]

        q_hybrid = (q_arima + resid_corr) * INFLOW_SCALING
    else:
        # ARIMA-only forecast
        q_hybrid = q_arima * INFLOW_SCALING

    q_hybrid = max(q_hybrid, 0.0)

    q_in_fore_values.append(q_hybrid)
    qin_combined.loc[t] = q_hybrid  # append so lags can see this

Q_in_fore = pd.Series(q_in_fore_values, index=forecast_index, name="Q_in_fore")
Q_in_fore_smooth = Q_in_fore.rolling(3, center=True, min_periods=1).mean()

# ---- 5.4 CYCLIC RECOVERY <-> GENERATION SIMULATION ----
H_vals = []
Q_out_vals = []
mode_vals = []

H = last_H

# Start in recovery mode (plant OFF)
mode = "recovery"
generation_start_time = None

for t in forecast_index:
    q_in = max(Q_in_fore_smooth.loc[t], 0.0)

    if mode == "recovery":
        q_out = 0.0
        H_next = H + (q_in - q_out) * 3600.0 / A_eff

        if H_next >= H_MAX_TARGET:
            H_next = H_MAX_TARGET
            mode = "generation"
            if generation_start_time is None:
                generation_start_time = t

    else:  # mode == "generation"
        q_out = QO_PER_MW * PLANNED_MW
        H_next = H + (q_in - q_out) * 3600.0 / A_eff

        if H_next <= H_MIN:
            H_next = H_MIN
            mode = "recovery"
            q_out = 0.0

    H_vals.append(H_next)
    Q_out_vals.append(q_out)
    mode_vals.append(mode)
    H = H_next

H_fore = pd.Series(H_vals, index=forecast_index, name="dam_el_fore")

result = pd.DataFrame({
    "rain_mm":       rain_fore.reindex(forecast_index).fillna(0.0),
    "Q_in_cms":      Q_in_fore_smooth,
    "Q_out_cms":     Q_out_vals,
    "dam_el_masl":   H_fore,
    "MW_schedule":   [0.0 if m == "recovery" else PLANNED_MW for m in mode_vals],
    "operation_mode": mode_vals,
})

result["dam_el_delta"] = result["dam_el_masl"].diff()
result["is_recovering"] = result["dam_el_delta"] > 0

total_hours = len(result)
recovery_hours = int(result["is_recovering"].sum())
max_recovery = float(result["dam_el_delta"].max(skipna=True) or 0.0)
net_change = float(result["dam_el_masl"].iloc[-1] - result["dam_el_masl"].iloc[0])

print("\n=== RECOVERY & GENERATION SUMMARY (7-DAY FORECAST) ===")
print(f"Total forecast hours      : {total_hours}")
print(f"Hours with level recovery : {recovery_hours}")
print(f"Max hourly recovery       : {max_recovery:.3f} m/h")
print(f"Net change in elevation   : {net_change:.3f} m over {total_hours} hours")

if generation_start_time is not None:
    hours_to_full = (generation_start_time - forecast_index[0]).total_seconds() / 3600.0
    print(f"First generation starts at: {generation_start_time} "
          f"({hours_to_full:.1f} hours from forecast start)")
else:
    print("Generation never started within forecast horizon (H_max not reached).")

result.to_csv(FORECAST_CSV)
print(f"\nSaved 7-day forecast to: {FORECAST_CSV}")
print(result.head())

if DO_PLOT:
    plt.figure(figsize=(12, 6))

    # Historical dam elevation (last 7 days)
    df["dam_el"].tail(7*24).plot(label="Historical dam_el")

    # Forecasted dam elevation
    H_fore.plot(label="Forecast dam_el", linestyle="--")

    rec_points = result[result["operation_mode"] == "recovery"]
    gen_points = result[result["operation_mode"] == "generation"]

    if not rec_points.empty:
        plt.scatter(
            rec_points.index,
            rec_points["dam_el_masl"],
            marker="o",
            s=20,
            label="Recovery (plant OFF)"
        )

    if not gen_points.empty:
        plt.scatter(
            gen_points.index,
            gen_points["dam_el_masl"],
            marker="x",
            s=20,
            label=f"Generation (plant {PLANNED_MW:.1f} MW)"
        )

    plt.axhline(H_MIN, color="red", linestyle=":", label=f"H_min={H_MIN}")
    plt.axhline(H_MAX_TARGET, color="green", linestyle=":", label=f"H_max target={H_MAX_TARGET}")

    if generation_start_time is not None:
        plt.axvline(generation_start_time, color="purple", linestyle="--",
                    label="First generation start")

    plt.title("7-Day Dam Elevation Simulation (Hybrid ARIMA+ANN)\n(Cyclic Recovery ↔ Generation)")
    plt.xlabel("Date")
    plt.ylabel("Elevation (masl)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
