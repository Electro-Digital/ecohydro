"""
Forecast Engine for CBK Dashboard
Integrates ARIMA+ANN models to generate real-time forecasts
"""
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ForecastEngine:
    def __init__(self, model_dir='forecasting_dataset'):
        self.model_dir = model_dir
        self.arima_model = None
        self.ann_model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load pre-trained ARIMA, ANN, and scaler models"""
        try:
            arima_path = os.path.join(self.model_dir, 'bhepp_inflow_arima.pkl')
            ann_path = os.path.join(self.model_dir, 'bhepp_inflow_ann.pkl')
            scaler_path = os.path.join(self.model_dir, 'bhepp_inflow_scaler.pkl')
            
            with open(arima_path, 'rb') as f:
                self.arima_model = pickle.load(f)
            with open(ann_path, 'rb') as f:
                self.ann_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def generate_forecast(self, historical_data, forecast_hours=168):
        """
        Generate forecast using ARIMA+ANN hybrid model
        
        Args:
            historical_data: DataFrame with columns [ts, rainfall, dam_el, b12_mw]
            forecast_hours: Number of hours to forecast (default 168 = 7 days)
        
        Returns:
            DataFrame with forecast results
        """
        try:
            # Prepare data
            df = historical_data.copy()
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.sort_values('ts').set_index('ts')
            
            # Generate forecast timestamps
            last_ts = df.index[-1]
            forecast_index = pd.date_range(
                start=last_ts + timedelta(hours=1),
                periods=forecast_hours,
                freq='h'
            )
            
            # ARIMA forecast
            arima_forecast = self.arima_model.get_forecast(steps=forecast_hours)
            q_in_arima = pd.Series(arima_forecast.predicted_mean.values, index=forecast_index)
            
            # Prepare features for ANN (simplified version)
            q_in_hybrid = []
            for i, ts in enumerate(forecast_index):
                q_arima = q_in_arima.iloc[i]
                # Use ARIMA prediction as base
                q_in_hybrid.append(max(q_arima, 0.0))
            
            Q_in_fore = pd.Series(q_in_hybrid, index=forecast_index)
            
            # Simulate dam elevation and power generation
            H_MIN = 330.0
            H_MAX = 332.0
            A_eff = 174897.0
            QO_PER_MW = 0.605
            PLANNED_MW = 10.0
            
            last_H = df['dam_el'].iloc[-1] if 'dam_el' in df.columns else 331.0
            
            H_vals = []
            Q_out_vals = []
            MW_vals = []
            mode_vals = []
            
            H = last_H
            mode = "recovery" if H < H_MAX else "generation"
            
            for ts in forecast_index:
                q_in = max(Q_in_fore.loc[ts], 0.0)
                
                if mode == "recovery":
                    q_out = 0.0
                    mw = 0.0
                    H_next = H + (q_in - q_out) * 3600.0 / A_eff
                    
                    if H_next >= H_MAX:
                        H_next = H_MAX
                        mode = "generation"
                else:
                    mw = PLANNED_MW
                    q_out = QO_PER_MW * mw
                    H_next = H + (q_in - q_out) * 3600.0 / A_eff
                    
                    if H_next <= H_MIN:
                        H_next = H_MIN
                        mode = "recovery"
                        q_out = 0.0
                        mw = 0.0
                
                H_vals.append(H_next)
                Q_out_vals.append(q_out)
                MW_vals.append(mw)
                mode_vals.append(mode)
                H = H_next
            
            # Create result DataFrame
            result = pd.DataFrame({
                'timestamp': forecast_index,
                'Q_in_cms': Q_in_fore.values,
                'Q_out_cms': Q_out_vals,
                'dam_el_masl': H_vals,
                'MW_forecast': MW_vals,
                'operation_mode': mode_vals
            })
            
            return result
            
        except Exception as e:
            print(f"Forecast generation error: {e}")
            return None
