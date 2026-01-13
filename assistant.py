import streamlit as st
import google.generativeai as genai
from datetime import datetime
from typing import Optional, Dict, Any

class GeminiHealthChatbot:
    """
    Advanced Gemini-based health assistant dengan:
    - Memori percakapan (chat history)
    - Analisis data sensor real-time
    - Empati & ketegasan berbasis risiko
    - Dialog natural dan kontekstual
    - Prioritas keselamatan pasien
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.chat_session = None
        self._configure_api()
        self._initialize_chat()

    def _configure_api(self):
        """Konfigurasi API Key dari st.secrets atau environment"""
        api_key = None
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
            else:
                import os
                api_key = os.environ.get("GOOGLE_API_KEY")
        except:
            pass

        if not api_key:
            st.error("GOOGLE_API_KEY tidak ditemukan di secrets.toml atau environment variable.")
            self.ready = False
            return

        try:
            genai.configure(api_key=api_key)
            self.ready = True
        except Exception as e:
            st.error(f"Gagal menginisialisasi Gemini: {e}")
            self.ready = False

    def _initialize_chat(self):
        """Inisialisasi sesi chat dengan system prompt yang kuat"""
        if not self.ready:
            return

        system_prompt = """
        Kamu adalah **Asisten Kesehatan Rumah Pintar** yang cerdas, empati, dan sangat berhati-hati.
        
        PEDOMAN UTAMA:
        - SELALU prioritaskan keselamatan pengguna.
        - Jangan pernah mendiagnosis penyakit atau meresepkan obat.
        - Jika ada indikasi kondisi serius (sesak napas, nyeri dada, pingsan, demam tinggi >39°C, dll), 
          TEGAS sarankan segera ke IGD atau hubungi 119.
        - Gunakan bahasa yang hangat, empati, tapi tegas saat diperlukan.
        - Analisis data sensor (suhu, kelembapan, gas) dan beri interpretasi sederhana yang membantu.
        - Ingat dan rujuk kembali percakapan sebelumnya jika relevan.
        - Jawab secara alami seperti manusia yang peduli, bukan robot kaku.
        - Untuk tambahan informasi, anda bisa mengakses halodoc, WHO, atau lembaga kesehatan terpercaya lainnya.
        - Berikan beberapa tips pertolongan pertama dan saran praktis

        Gaya komunikasi:
        - Gunakan bahasa Indonesia yang ramah dan mudah dipahami.
        - Beri saran praktis untuk kenyamanan sehari-hari.
        - Jika data sensor abnormal, beri peringatan dini dengan nada prihatin tapi tidak menakutkan.
        """

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            self.chat_session = model.start_chat()
        except Exception as e:
            st.error(f"Gagal memulai sesi chat: {e}")
            self.ready = False

    def ask(self, user_message: str, sensor_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Kirim pertanyaan pengguna ke Gemini dengan konteks sensor terkini.
        
        Args:
            user_message: Pesan dari pengguna
            sensor_context: Dict berisi data sensor terbaru (temp, hum, gas, ai, ts, device)
        
        Returns:
            Jawaban dari Gemini
        """
        if not self.ready or not self.chat_session:
            return "Maaf, asisten kesehatan belum siap. Periksa konfigurasi API key."

        # Bangun konteks sensor yang informatif
        context_lines = ["**Data Sensor Terkini:**"]
        if sensor_context:
            ts = sensor_context.get("ts", "tidak diketahui")
            if isinstance(ts, datetime):
                ts = ts.strftime("%d %b %Y, %H:%M")
            elif isinstance(ts, str):
                try:
                    ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %H:%M")
                except:
                    pass

            context_lines.extend([
                f"Waktu: {ts}",
                f"Suhu ruangan: {sensor_context.get('temp', '?'):.1f}°C",
                f"Kelembapan: {sensor_context.get('hum', '?'):.1f}%",
                f"Kadar gas (ppm): {sensor_context.get('gas', '?'):.0f}",
                f"Status AI: {sensor_context.get('ai', 'N/A')}",
            ])

            # Interpretasi otomatis sederhana
            temp_val = sensor_context.get('temp', 0)
            hum_val = sensor_context.get('hum', 0)
            gas_val = sensor_context.get('gas', 0)

            if temp_val > 32:
                context_lines.append("Suhu ruangan cukup panas – pastikan hidrasi dan ventilasi baik.")
            elif temp_val < 18:
                context_lines.append("Suhu ruangan dingin – gunakan pakaian hangat jika perlu.")

            if gas_val > 800:
                context_lines.append("PERINGATAN: Kadar gas tinggi! Segera buka jendela dan periksa sumbernya.")
            elif gas_val > 400:
                context_lines.append("Kadar gas mulai tinggi – perhatikan ventilasi ruangan.")

        else:
            context_lines.append("Data sensor belum tersedia.")

        full_context = "\n".join(context_lines)
        
        try:
            response = self.chat_session.send_message(
                f"{full_context}\n\nPertanyaan pengguna:\n{user_message}"
            )
            return response.text.strip()
        except Exception as e:
            return f"Maaf, terjadi kesalahan saat berkomunikasi dengan asisten: {str(e)}"