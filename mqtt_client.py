import json
import threading
import os
from datetime import datetime
import pandas as pd
import paho.mqtt.client as mqtt
from model import ModelService

TOPIC_DATA = "SHHE/data"
TOPIC_STATUS = "SHHE/status"
TOPIC_OBAT = "SHHE/obat"

class MQTTRunner:
    def __init__(self, broker, port, model_path="models/smarthealth.retrained.pkl", csv_path="data.csv"):
        self.broker = broker
        self.port = port
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.lock = threading.Lock()
        self.last_status = "N/A"
        self.latest_record = None
        self.last_timestamp = {}

        # load model
        # baru
        # mqtt_client.py bagian __init__
        self.model = None
        if model_path:  # tetap pakai model_path sebagai argumen
           try:
        # panggil ModelService dengan model_source, bisa path atau dict
                  self.model = ModelService(model_source=model_path)
           except Exception as e:
                  print("[MQTT] Warning: Failed to load model:", e)


        self.csv_path = csv_path
        try:
            pd.read_csv(self.csv_path)
        except Exception:
            df = pd.DataFrame(columns=["ts", "device", "temp", "hum", "gas", "ai", "heartrate"])
            df.to_csv(self.csv_path, index=False)

    def _on_connect(self, client, userdata, flags, rc):
        print("[MQTT] Connected, subscribing ...")
        client.subscribe(TOPIC_DATA)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            device = payload.get("device", "Smart Home Health Ecosystem")
            ts = payload.get("ts")
            if not ts:
                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            temp = float(payload.get("temp", 0.0))
            hum = float(payload.get("hum", 0.0))
            gas = float(payload.get("gas", 0.0))

            heartrate = payload.get("heartrate")
            try:
                heartrate = float(heartrate) if heartrate is not None else 0.0  # default awal 0
            except:
                heartrate = 0.0

# VALIDASI HEARTRATE
            if heartrate > 220 or heartrate < 30:
                heartrate = 0


            # AI prediction
            label = "GOOD"
            if self.model is not None:
                try:
                    if hasattr(self.model, "predict_from_features"):
                        last_ts = self.last_timestamp.get(device)
                        if last_ts and ts <= last_ts:
                            return   # drop packet lama / duplicate

                        self.last_timestamp[device] = ts
                        features = self.model.compute_features(device, temp, hum, gas, ts, heartrate)
                        label = self.model.predict_from_features(features)
                    else:
                        print("[MQTT] Warning: self.model bukan ModelService, skipping AI prediction")
                except Exception as e:
                    print("[MQTT] AI prediction error:", e)

            # CSV
            row = {"ts": ts, "device": device, "temp": temp, "hum": hum,
                   "gas": gas, "ai": label, "heartrate": heartrate}
            self._append_csv(row)

            # Publish status
            if label != self.last_status:
                out = {"status": label}
                client.publish(TOPIC_STATUS, json.dumps(out))

            with self.lock:
                self.last_status = label
                self.latest_record = row

            print(f"[MQTT] {device} {ts} => T:{temp}°C H:{hum}% G:{gas} HR:{heartrate}BPM => {label}")

        except Exception as e:
            print("[MQTT] on_message error:", e)

    def _append_csv(self, row):
        with self.lock:
            try:
                df = pd.read_csv(self.csv_path)
            except:
                df = pd.DataFrame(columns=row.keys())

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        tmp = self.csv_path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.csv_path)


    def start(self):
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        try:
            self.client.connect(self.broker, self.port, 60)
        except Exception as e:
            print("[MQTT] Connect failed:", e)
            return
        self.client.loop_forever()

    def publish_obat(self, schedules):
        if schedules:
            payload = {"schedules": schedules}
            self.client.publish(TOPIC_OBAT, json.dumps(payload))
            print("[MQTT] Published schedules:", schedules)

    def get_last_status(self):
        with self.lock:
            return self.last_status

    def get_latest_record(self):
        with self.lock:
            return self.latest_record

    def get_csv_path(self):
        return self.csv_path
