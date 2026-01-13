import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
from assistant import GeminiHealthChatbot

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="Smart Health Ecosystem",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============= LOAD CUSTOM CSS FROM EXTERNAL FILE =============
def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ============= CONFIG & INIT =============
BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
PORT = int(st.secrets.get("MQTT_PORT", 1883))
MODEL_PATH = "models/smarthealth_retrained.pkl"
CSV_PATH = "data.csv"

from mqtt_client import MQTTRunner
if "mqtt_runner" not in st.session_state:
    runner = MQTTRunner(
        broker=BROKER,
        port=PORT,
        model_path=MODEL_PATH,
        csv_path=CSV_PATH
    )
    runner.start()
    st.session_state.mqtt_runner = runner


if "mqtt_runner" not in st.session_state:
    from mqtt_client import MQTTRunner
    st.session_state.mqtt_runner = MQTTRunner()
    st.session_state.mqtt_runner.start()

if not os.path.exists(MODEL_PATH):
    st.warning(f"Model tidak ditemukan di {MODEL_PATH}. Menjalankan mode terbatas (prediksi AI dinonaktifkan).")
    MODEL_AVAILABLE = False
else:
    MODEL_AVAILABLE = True

df = pd.read_csv("data.csv")

if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["ts", "device", "temp", "hum", "gas", "ai", "heartrate"]).to_csv(CSV_PATH, index=False)

if "medicine_schedules" not in st.session_state:
    st.session_state.medicine_schedules = []

# ============= LOAD DATA =============
expected_cols = ["ts", "device", "temp", "hum", "gas", "ai", "heartrate"]

def _safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        if not set(expected_cols).issubset(df.columns):
            df2 = pd.read_csv(path, header=None)
            if df2.shape[1] >= len(expected_cols):
                df2 = df2.iloc[:, :len(expected_cols)]
                df2.columns = expected_cols
                return df2
            return pd.DataFrame(columns=expected_cols)
        return df
    except Exception as e:
        print("Warning reading CSV:", e)
        return pd.DataFrame(columns=expected_cols)

df = _safe_read_csv(CSV_PATH)
for col in ("temp", "hum", "gas", "heartrate"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        df[col] = 0

if "ts" in df.columns:
    try:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    except:
        df["ts"] = df["ts"].astype(str)

last_record = st.session_state.mqtt_runner.get_latest_record() if "mqtt_runner" in st.session_state else {}

if not last_record and not df.empty:
    last_row = df.iloc[-1].to_dict()
    last_record = {
        "ts": last_row.get("ts", ""),
        "device": last_row.get("device", ""),
        "temp": float(last_row.get("temp") or 0),
        "hum": float(last_row.get("hum") or 0),
        "gas": float(last_row.get("gas") or 0),
        "heartrate": float(last_row.get("heartrate") or 0) if last_row.get("heartrate") else 0,
        "ai": last_row.get("ai", "N/A")
    }

temp = float(last_record.get("temp", 0) or 0)
hum = float(last_record.get("hum", 0) or 0)
gas = float(last_record.get("gas", 0) or 0)
hr_raw = last_record.get("heartrate")
# hanya pakai heartrate jika >1, selain itu set 0
heartrate = float(hr_raw) if hr_raw and float(hr_raw) > 1 else 0
ai_status = last_record.get("ai", "N/A")

# ============= HEADER =============
st.markdown("<h1 class='dashboard-title'>🌡️ Smart Health Ecosystem</h1>", unsafe_allow_html=True)
st.markdown("<p class='dashboard-subtitle'>Real-time Health Monitoring dengan AI & IoT</p>", unsafe_allow_html=True)

# ============= TABS =============
tab_monitoring, tab_medicine = st.tabs(["Monitoring", "Medicine Scheduler"])

# ========== TAB 1: MONITORING (FULL WIDTH) =========
with tab_monitoring:
    st.markdown("<div class='section-header'>Live Sensor Metrics</div>", unsafe_allow_html=True)
    
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f"<div class='metric-card-modern'><div class='metric-label-modern'>Temperature</div><div class='metric-value-modern'>{temp:.1f}°C</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card-modern'><div class='metric-label-modern'>Humidity</div><div class='metric-value-modern'>{hum:.1f}%</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card-modern'><div class='metric-label-modern'>Gas Level</div><div class='metric-value-modern'>{gas:.0f}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card-modern'><div class='metric-label-modern'>Heart Rate</div><div class='metric-value-modern'>{heartrate:.0f}</div></div>", unsafe_allow_html=True)
    with col5:
        status_class = "status-normal" if ai_status == "Normal" else "status-alert" if ai_status == "Warning" else "status-danger"
        st.markdown(f"<div class='metric-card-modern'><div class='metric-label-modern'>AI Status</div><div class='metric-value-modern' style='font-size: 0.5rem;'><span class='status-badge'>{ai_status}</span></div></div>", unsafe_allow_html=True)
    with col6:
        if st.button("AUTO REFRESH", use_container_width=True, key="toggle_auto_refresh"):
            st.session_state.auto_refresh = not st.session_state.auto_refresh
    
    # ============= SENSOR VISUALIZATION =============
    st.markdown("<div class='section-header'>Visualisasi Data Sensor</div>", unsafe_allow_html=True)
    st.markdown("<div class='gauge-viz-container'>", unsafe_allow_html=True)
    
    col_gauge1, col_gauge2, col_gauge3, col_gauge4 = st.columns(4)
    with col_gauge1:
        temp_percent = min(100, max(0, (temp / 50) * 100))
        st.markdown(f"""
        <div class='gauge-circular-container'>
            <div class='gauge-circular-label'>Temperature</div>
            <div class='gauge-circular-wrapper'>
                <div class='gauge-circular-bg'></div>
                <div class='gauge-circular-fill' style='--gauge-percent: {temp_percent}%'></div>
                <div class='gauge-circular-text'>
                    <div class='gauge-circular-value'>{temp:.1f}</div>
                    <div class='gauge-circular-unit'>°C</div>
                </div>
            </div>
            <div class='gauge-stats-modern'>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Min</div><div class='gauge-stat-value-modern'>{df['temp'].min() if not df.empty else 0:.1f}°C</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Max</div><div class='gauge-stat-value-modern'>{df['temp'].max() if not df.empty else 0:.1f}°C</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Avg</div><div class='gauge-stat-value-modern'>{df['temp'].mean() if not df.empty else 0:.1f}°C</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Status</div><div class='gauge-stat-value-modern'>Optimal</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_gauge2:
        hum_percent = min(100, max(0, hum))
        st.markdown(f"""
        <div class='gauge-circular-container'>
            <div class='gauge-circular-label'>Humidity</div>
            <div class='gauge-circular-wrapper'>
                <div class='gauge-circular-bg'></div>
                <div class='gauge-circular-fill' style='--gauge-percent: {hum_percent}%'></div>
                <div class='gauge-circular-text'>
                    <div class='gauge-circular-value'>{hum:.1f}</div>
                    <div class='gauge-circular-unit'>%</div>
                </div>
            </div>
            <div class='gauge-stats-modern'>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Min</div><div class='gauge-stat-value-modern'>{df['hum'].min() if not df.empty else 0:.1f}%</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Max</div><div class='gauge-stat-value-modern'>{df['hum'].max() if not df.empty else 0:.1f}%</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Avg</div><div class='gauge-stat-value-modern'>{df['hum'].mean() if not df.empty else 0:.1f}%</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Status</div><div class='gauge-stat-value-modern'>Good</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_gauge3:
        gas_percent = min(100, max(0, (gas / 1000) * 100))
        st.markdown(f"""
        <div class='gauge-circular-container'>
            <div class='gauge-circular-label'>Gas Level</div>
            <div class='gauge-circular-wrapper'>
                <div class='gauge-circular-bg'></div>
                <div class='gauge-circular-fill' style='--gauge-percent: {gas_percent}%'></div>
                <div class='gauge-circular-text'>
                    <div class='gauge-circular-value'>{gas:.0f}</div>
                    <div class='gauge-circular-unit'>ppm</div>
                </div>
            </div>
            <div class='gauge-stats-modern'>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Min</div><div class='gauge-stat-value-modern'>{df['gas'].min() if not df.empty else 0:.0f}</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Max</div><div class='gauge-stat-value-modern'>{df['gas'].max() if not df.empty else 0:.0f}</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Avg</div><div class='gauge-stat-value-modern'>{df['gas'].mean() if not df.empty else 0:.0f}</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Status</div><div class='gauge-stat-value-modern'>Safe</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_gauge4:
        if heartrate > 0:
            hr_percent = min(100, max(0, (heartrate / 200) * 100))
            hr_display_status = "Normal" if 60 <= heartrate <= 100 else "Elevated" if heartrate > 100 else "Low"
        else:
            hr_percent = 0
            hr_display_status = "N/A"
        st.markdown(f"""
        <div class='gauge-circular-container'>
            <div class='gauge-circular-label'>Heart Rate</div>
            <div class='gauge-circular-wrapper'>
                <div class='gauge-circular-bg'></div>
                <div class='gauge-circular-fill' style='--gauge-percent: {hr_percent}%'></div>
                <div class='gauge-circular-text'>
                    <div class='gauge-circular-value'>{heartrate:.0f}</div>
                    <div class='gauge-circular-unit'>BPM</div>
                </div>
            </div>
            <div class='gauge-stats-modern'>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Min</div><div class='gauge-stat-value-modern'>{df['heartrate'].min() if not df.empty and 'heartrate' in df.columns else 0:.0f}</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Max</div><div class='gauge-stat-value-modern'>{df['heartrate'].max() if not df.empty and 'heartrate' in df.columns else 0:.0f}</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Avg</div><div class='gauge-stat-value-modern'>{df['heartrate'].mean() if not df.empty and 'heartrate' in df.columns else 0:.0f}</div></div>
                <div class='gauge-stat-modern'><div class='gauge-stat-label-modern'>Status</div><div class='gauge-stat-value-modern'>{hr_display_status}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ============= TREND CHART =============
    st.markdown("<div class='section-header'>Tren Grafik Data Lingkungan</div>", unsafe_allow_html=True)
    st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
    if not df.empty:
        recent = df.tail(200).copy()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=recent['ts'], y=recent['temp'], name='Temperature', line=dict(color='#ff6b6b', width=3), mode='lines', fill='tonexty', fillcolor='rgba(255, 107, 107, 0.1)', yaxis='y1'))
        fig_trend.add_trace(go.Scatter(x=recent['ts'], y=recent['hum'], name='Humidity', line=dict(color='#1db8a0', width=3), mode='lines', fill='tonexty', fillcolor='rgba(29, 184, 160, 0.1)', yaxis='y2'))
        fig_trend.add_trace(go.Scatter(x=recent['ts'], y=recent['gas']/10, name='Gas (÷10)', line=dict(color='#2dd9ce', width=3), mode='lines', fill='tonexty', fillcolor='rgba(45, 217, 206, 0.1)', yaxis='y3'))
        if 'heartrate' in recent.columns:
            fig_trend.add_trace(go.Scatter(x=recent['ts'], y=recent['heartrate'], name='Heart Rate', line=dict(color='#f44336', width=3), mode='lines+markers', yaxis='y4'))
        fig_trend.update_layout(
            hovermode='x unified', 
            plot_bgcolor='rgba(15, 31, 30, 0.5)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            xaxis=dict(
                showgrid=True, 
                gridcolor='rgba(29, 184, 160, 0.15)', 
                color='#2dd9ce',
                tickformat='%Y-%m-%d %H:%M:%S'
            ), 
            height=320, 
            margin=dict(l=60, r=30, t=30, b=60),
            font={'family': 'Poppins', 'color': '#2dd9ce', 'size': 11},
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, bgcolor='rgba(25, 35, 33, 0.9)', bordercolor='rgba(29, 184, 160, 0.25)', borderwidth=2),
            yaxis=dict(showgrid=True, gridcolor='rgba(29, 184, 160, 0.15)', color='#2dd9ce', title='Temp/Humidity/Gas'),
            yaxis2=dict(showgrid=False, color='#2dd9ce', overlaying='y', side='right'),
            yaxis3=dict(showgrid=False, color='#2dd9ce', overlaying='y', side='right'),
            yaxis4=dict(showgrid=False, color='#f44336', overlaying='y', side='right')
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Menunggu data sensor...")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ============= HEALTH ASSISTANT =============
    st.markdown("<div class='section-header'>Asisten Kesehatan</div>", unsafe_allow_html=True)

    if "health_chatbot" not in st.session_state:
        with st.spinner("Menghubungkan ke Asisten Kesehatan Gemini"):
            st.session_state.health_chatbot = GeminiHealthChatbot(model_name="gemini-2.5-flash")

    chatbot = st.session_state.health_chatbot

    if not getattr(chatbot, "ready", False):
        st.warning("Asisten belum siap. Pastikan GOOGLE_API_KEY ada di .streamlit/secrets.toml")
    else:
        st.markdown("<div class='modern-card'>", unsafe_allow_html=True)

        user_input = st.text_area(
            "Tanyakan apa saja: gejala kesehatan, interpretasi sensor, saran harian, dll",
            height=150,
            key="health_chat_input_full",
            placeholder="Contoh: Saya merasa pusing, suhu ruangan 32°C, kelembapan tinggi, dan ada bau aneh. Apa yang harus saya lakukan?"
        )

        if st.button("Kirim ke Asisten Kesehatan", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("Asisten sedang menganalisis data sensor dan pertanyaan Anda..."):
                    reply = chatbot.ask(user_input, sensor_context=last_record)
                    st.markdown(f"""
                    <div class='info-card-modern'>
                        <strong>Jawaban dari Asisten Kesehatan:</strong><br><br>
                        {reply}
                    </div>
                    """, unsafe_allow_html=True)

        
    
    # ============= AUTO REFRESH LOGIC =============
    if st.session_state.auto_refresh:
        time.sleep(1)
        st.rerun()
    
    # ============= FOOTER =============
    st.markdown("<div class='footer-card'><p style='color: #2dd9ce; font-size: 0.85rem; margin: 0; font-weight: 700;'> Smart Health Ecosystem © 2025 | Real-time Monitoring System </p></div>", unsafe_allow_html=True)

# ========== TAB 2: MEDICINE SCHEDULER (FULL WIDTH) =========
with tab_medicine:
    # ============= JADWAL MINUM OBAT (3 FORMS) =============
    st.markdown("<h2 style='color: #2dd9ce; font-family: Space Grotesk; font-weight: 700; margin: 2rem 0 1.5rem 0;'>Jadwal Minum Obat</h2>", unsafe_allow_html=True)
    
    # Buat 3 kolom untuk 3 form jadwal
    col_form1, col_form2, col_form3 = st.columns(3)
    
    # FORM 1
    with col_form1:
        st.markdown("<h3 style='color: #26d0ce; font-size: 1rem; margin-bottom: 1rem;'>Jadwal 1</h3>", unsafe_allow_html=True)
        
        medicine_name_1 = st.text_input("Nama Obat", "", key="med_name_1", placeholder="Paracetamol")
        
        col_dates_1 = st.columns(2)
        with col_dates_1[0]:
            start_date_1 = st.date_input("Tgl Mulai", datetime.now(), key="start_date_1")
        with col_dates_1[1]:
            end_date_1 = st.date_input("Tgl Selesai", datetime.now() + timedelta(days=7), key="end_date_1")
        
        frequency_1 = st.number_input("Frekuensi", min_value=1, max_value=6, value=2, key="freq_1")
        
        st.markdown("<p style='color: #26d0ce; font-size: 0.8rem; font-weight: 700; margin: 0.8rem 0 0.5rem 0;'>Waktu:</p>", unsafe_allow_html=True)
        times_1 = []
        cols_times_1 = st.columns(min(frequency_1, 3))
        for i in range(frequency_1):
            with cols_times_1[i % 3]:
                time_input = st.time_input(f"Jam {i+1}", datetime.strptime(f"{8 + i*4}:00", "%H:%M").time(), key=f"time_1_{i}")
                times_1.append(time_input)
        
        if st.button("TAMBAH", key="add_schedule_1", use_container_width=True):
            if medicine_name_1.strip() and start_date_1 and end_date_1 and times_1:
                if start_date_1 <= end_date_1:
                    schedules = []
                    current_date = start_date_1
                    while current_date <= end_date_1:
                        for t in times_1:
                            schedule_datetime = datetime.combine(current_date, t)
                            schedules.append(schedule_datetime.strftime("%Y-%m-%d %H:%M"))
                        current_date += timedelta(days=1)
                    
                    schedules = list(dict.fromkeys(schedules))
                    existing_datetimes = set([s["datetime"] for s in st.session_state.medicine_schedules])
                    new_schedules = [{"datetime": s, "medicine": medicine_name_1} for s in schedules if s not in existing_datetimes]
                    
                    if new_schedules:
                        st.session_state.medicine_schedules.extend(new_schedules)
                        new_datetimes = [s["datetime"] for s in new_schedules]
                        if "mqtt_runner" in st.session_state:
                            st.session_state.mqtt_runner.publish_obat(new_datetimes)

                        st.success(f"Jadwal 1 ditambahkan: {len(new_schedules)} jadwal baru!")
                    else:
                        st.warning("Semua jadwal sudah ada dalam daftar.")
                else:
                    st.error("Tanggal mulai harus <= tanggal selesai!")
            else:
                st.error("Isi semua field!")
        
    # FORM 2
    with col_form2:
        st.markdown("<h3 style='color: #26d0ce; font-size: 1rem; margin-bottom: 1rem;'>Jadwal 2</h3>", unsafe_allow_html=True)
        
        medicine_name_2 = st.text_input("Nama Obat", "", key="med_name_2", placeholder="Amoxicillin")
        
        col_dates_2 = st.columns(2)
        with col_dates_2[0]:
            start_date_2 = st.date_input("Tgl Mulai", datetime.now(), key="start_date_2")
        with col_dates_2[1]:
            end_date_2 = st.date_input("Tgl Selesai", datetime.now() + timedelta(days=7), key="end_date_2")
        
        frequency_2 = st.number_input("Frekuensi", min_value=1, max_value=6, value=3, key="freq_2")
        
        st.markdown("<p style='color: #26d0ce; font-size: 0.8rem; font-weight: 700; margin: 0.8rem 0 0.5rem 0;'>Waktu:</p>", unsafe_allow_html=True)
        times_2 = []
        cols_times_2 = st.columns(min(frequency_2, 3))
        for i in range(frequency_2):
            with cols_times_2[i % 3]:
                time_input = st.time_input(f"Jam {i+1}", datetime.strptime(f"{8 + i*4}:00", "%H:%M").time(), key=f"time_2_{i}")
                times_2.append(time_input)
        
        if st.button("TAMBAH", key="add_schedule_2", use_container_width=True):
            if medicine_name_2.strip() and start_date_2 and end_date_2 and times_2:
                if start_date_2 <= end_date_2:
                    schedules = []
                    current_date = start_date_2
                    while current_date <= end_date_2:
                        for t in times_2:
                            schedule_datetime = datetime.combine(current_date, t)
                            schedules.append(schedule_datetime.strftime("%Y-%m-%d %H:%M"))
                        current_date += timedelta(days=1)
                    
                    schedules = list(dict.fromkeys(schedules))
                    existing_datetimes = set([s["datetime"] for s in st.session_state.medicine_schedules])
                    new_schedules = [{"datetime": s, "medicine": medicine_name_2} for s in schedules if s not in existing_datetimes]
                    
                    if new_schedules:
                        st.session_state.medicine_schedules.extend(new_schedules)
                        new_datetimes = [s["datetime"] for s in new_schedules]
                        if "mqtt_runner" in st.session_state:
                            st.session_state.mqtt_runner.publish_obat(new_datetimes)

                        st.success(f"Jadwal 2 ditambahkan: {len(new_schedules)} jadwal baru!")
                    else:
                        st.warning("Semua jadwal sudah ada dalam daftar.")
                else:
                    st.error("Tanggal mulai harus <= tanggal selesai!")
            else:
                st.error("Isi semua field!")
        
    # FORM 3
    with col_form3:
        st.markdown("<h3 style='color: #26d0ce; font-size: 1rem; margin-bottom: 1rem;'>Jadwal 3</h3>", unsafe_allow_html=True)
        
        medicine_name_3 = st.text_input("Nama Obat", "", key="med_name_3", placeholder="Vitamin C")
        
        col_dates_3 = st.columns(2)
        with col_dates_3[0]:
            start_date_3 = st.date_input("Tgl Mulai", datetime.now(), key="start_date_3")
        with col_dates_3[1]:
            end_date_3 = st.date_input("Tgl Selesai", datetime.now() + timedelta(days=7), key="end_date_3")
        
        frequency_3 = st.number_input("Frekuensi", min_value=1, max_value=6, value=1, key="freq_3")
        
        st.markdown("<p style='color: #26d0ce; font-size: 0.8rem; font-weight: 700; margin: 0.8rem 0 0.5rem 0;'>Waktu:</p>", unsafe_allow_html=True)
        times_3 = []
        cols_times_3 = st.columns(min(frequency_3, 3))
        for i in range(frequency_3):
            with cols_times_3[i % 3]:
                time_input = st.time_input(f"Jam {i+1}", datetime.strptime(f"{8 + i*4}:00", "%H:%M").time(), key=f"time_3_{i}")
                times_3.append(time_input)
        
        if st.button("TAMBAH", key="add_schedule_3", use_container_width=True):
            if medicine_name_3.strip() and start_date_3 and end_date_3 and times_3:
                if start_date_3 <= end_date_3:
                    schedules = []
                    current_date = start_date_3
                    while current_date <= end_date_3:
                        for t in times_3:
                            schedule_datetime = datetime.combine(current_date, t)
                            schedules.append(schedule_datetime.strftime("%Y-%m-%d %H:%M"))
                        current_date += timedelta(days=1)
                    
                    schedules = list(dict.fromkeys(schedules))
                    existing_datetimes = set([s["datetime"] for s in st.session_state.medicine_schedules])
                    new_schedules = [{"datetime": s, "medicine": medicine_name_3} for s in schedules if s not in existing_datetimes]
                    
                    if new_schedules:
                        st.session_state.medicine_schedules.extend(new_schedules)
                        new_datetimes = [s["datetime"] for s in new_schedules]
                        if "mqtt_runner" in st.session_state:
                            st.session_state.mqtt_runner.publish_obat(new_datetimes)

                        st.success(f"Jadwal 3 ditambahkan: {len(new_schedules)} jadwal baru!")
                    else:
                        st.warning("Semua jadwal sudah ada dalam daftar.")
                else:
                    st.error("Tanggal mulai harus <= tanggal selesai!")
            else:
                st.error("Isi semua field!")
        
    
    # ============= STATISTIK JADWAL =============
    st.markdown("<div class='section-header'>Statistik Jadwal</div>", unsafe_allow_html=True)
    st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
    
    total_schedules = len(st.session_state.medicine_schedules)
    unique_medicines = len(set([s["medicine"] for s in st.session_state.medicine_schedules])) if st.session_state.medicine_schedules else 0
    
    col_stat_1, col_stat_2, col_stat_3 = st.columns(3)
    
    with col_stat_1:
        st.markdown(f"""
        <div class='metric-card-modern'>
            <div class='metric-label-modern'>Total Jadwal</div>
            <div class='metric-value-modern'>{total_schedules}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat_2:
        st.markdown(f"""
        <div class='metric-card-modern'>
            <div class='metric-label-modern'>Jenis Obat</div>
            <div class='metric-value-modern'>{unique_medicines}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat_3:
        if st.button("Hapus Semua Jadwal", key="clear_schedules", use_container_width=True):
            st.session_state.medicine_schedules = []
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ============= SCHEDULE LIST =============
    if st.session_state.medicine_schedules:
        st.markdown("<div class='section-header'>Daftar Jadwal Obat</div>", unsafe_allow_html=True)
        st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
        
        schedule_df = pd.DataFrame(st.session_state.medicine_schedules)
        schedule_df = schedule_df.sort_values('datetime')
        
        for medicine in schedule_df['medicine'].unique():
            med_schedules = schedule_df[schedule_df['medicine'] == medicine]
            with st.expander(f"Obat: {medicine} ({len(med_schedules)} jadwal)", expanded=False):
                st.dataframe(med_schedules[['datetime', 'medicine']], use_container_width=True, hide_index=True, key=f"df_{medicine}")
        
        st.markdown("<p style='color: #26d0ce; font-size: 0.9rem; font-weight: 700; margin-top: 1.5rem;'>Ringkasan Semua Jadwal:</p>", unsafe_allow_html=True)
        st.dataframe(schedule_df, use_container_width=True, hide_index=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Belum ada jadwal obat. Silakan tambahkan jadwal baru di atas.")
    
    st.markdown("<div class='footer-card'><p style='color: #2dd9ce; font-size: 0.85rem; margin: 0; font-weight: 700;'> Smart Health Ecosystem © 2025 | Medicine Scheduler </p></div>", unsafe_allow_html=True)

# ============= AUTO REFRESH LOGIC (GLOBAL) =============
if st.session_state.auto_refresh:
    time.sleep(1)
    st.rerun()

time.sleep(0.1)