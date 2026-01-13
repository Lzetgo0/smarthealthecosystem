import joblib
import numpy as np
import pandas as pd
from collections import deque
import os

class ModelService:
    def __init__(self, model_source, roll_size=3):
        """
        model_source: str (pkl path) atau dict {'model':..., 'scaler':..., 'features':...}
        """
        if isinstance(model_source, str):
            import joblib, os
            if not os.path.exists(model_source):
                raise FileNotFoundError(f"Model not found at {model_source}")
            data = joblib.load(model_source)
        elif isinstance(model_source, dict):
            data = model_source
        else:
            raise ValueError("model_source must be a path or a dict")

        self.model = data["model"]
        self.scaler = data.get("scaler", None)
        self.features = data.get("features", None)

        # history per device
        from collections import deque
        self.history = {}
        self.last = {}
        self.roll_size = roll_size


    # ---------------- INTERNAL ----------------
    def _ensure_device(self, device):
        if device not in self.history:
            self.history[device] = {"temp": deque(maxlen=self.roll_size),
                                    "hum": deque(maxlen=self.roll_size),
                                    "gas": deque(maxlen=self.roll_size)}
            self.last[device] = {"temp": None, "hum": None, "gas": None}

    # ---------------- FEATURES ----------------
    def compute_features(self, device, temp, hum, gas, ts=None, heartrate=None):
        self._ensure_device(device)
        last = self.last[device]

        d_temp = 0.0 if last["temp"] is None else temp - last["temp"]
        d_hum  = 0.0 if last["hum"] is None else hum - last["hum"]
        d_gas  = 0.0 if last["gas"] is None else gas - last["gas"]

    # update history
        h = self.history[device]
        h["temp"].append(temp)
        h["hum"].append(hum)
        h["gas"].append(gas)

        r_temp = float(sum(h["temp"]) / len(h["temp"]))
        r_hum  = float(sum(h["hum"]) / len(h["hum"]))
        r_gas  = float(sum(h["gas"]) / len(h["gas"]))

    # ================== NEW: trend features ==================
        prev_avg = self.last.get(device + "_avg", None)
        if prev_avg is None:
            trend_temp = 0.0
            trend_gas  = 0.0
        else:
            trend_temp = r_temp - prev_avg["temp"]
            trend_gas  = r_gas  - prev_avg["gas"]

        self.last[device + "_avg"] = {"temp": r_temp, "gas": r_gas}
    # =========================================================

        self.last[device] = {"temp": temp, "hum": hum, "gas": gas}

        hr = float(heartrate) if heartrate is not None else 0.0

        features = np.array([[
            temp, hum, gas,
            d_temp, d_hum, d_gas,
            r_temp, r_hum, r_gas,
            hr,
            trend_temp, trend_gas   # ← dua fitur yang hilang
        ]])
        return features
    
    # ---------------- PREDICTION ----------------
    def predict_from_features(self, features):
        arr = np.asarray(features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

    # Build DataFrame for scaler (it needs feature names)
        feat_names = self.features
        if feat_names is None:
            feat_names = [
                "temp","hum","gas",
                "d_temp","d_hum","d_gas",
                "r_temp","r_hum","r_gas",
                "heartrate",
                "trend_temp","trend_gas"
            ]

        X_df = pd.DataFrame(arr, columns=feat_names)

    # kill NaN / inf from sensors & startup
        X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # apply scaler if exists (scaler wants DataFrame)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_df)
        else:
            X_scaled = X_df.values

    # RandomForest was trained WITHOUT feature names → give numpy
        X_final = np.asarray(X_scaled)

        pred = int(self.model.predict(X_final)[0])
        ai_label = {0:"GOOD",1:"ALERT",2:"DANGER"}.get(pred, "UNKNOWN")

    # ================= RULE ENGINE =================
        vals = X_final[0] if isinstance(X_final, np.ndarray) else X_final.values[0]

        temp, hum, gas, d_temp, d_hum, d_gas, r_temp, r_hum, r_gas, hr, trend_temp, trend_gas = vals


        rule_label = ai_label

    # ----- GAS rules -----
        if gas > 1200 or r_gas > 1000:
            rule_label = "DANGER" 
        elif gas > 700:
            rule_label = max(rule_label, "ALERT", key=["GOOD","ALERT","DANGER"].index)

    # ----- Temperature rules -----
        if temp > 38 or temp < 18:
            rule_label = "ALERT"

    # ----- Heart rate rules -----
        if hr > 140 or hr < 40:
            rule_label = "DANGER"
        elif hr > 110:
            rule_label = max(rule_label, "ALERT", key=["GOOD","ALERT","DANGER"].index)

    # ----- Trend danger -----
        if trend_gas > 80 or trend_temp > 2:
            rule_label = max(rule_label, "ALERT", key=["GOOD","ALERT","DANGER"].index)

    # ==============================================

        return rule_label

