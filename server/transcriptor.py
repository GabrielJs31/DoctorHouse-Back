from faster_whisper import WhisperModel
from tqdm import tqdm
import os, time

# ▌Carpeta de audio relativa
BASE_DIR = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ▌Ruta del archivo de entrada y salida
audio_path = os.path.join(AUDIO_DIR, "audio.wav")
output_path = os.path.splitext(audio_path)[0] + ".txt"  # audio/audio.txt

# ▌CONFIGURACIÓN DEL MODELO SOLO CPU
model = WhisperModel("small", device="cpu", compute_type="int8")

# ▌TIEMPO DE EJECUCIÓN
start_time = time.time()

# ▌TRANSCRIPCIÓN
segments, info = model.transcribe(audio_path, beam_size=1, vad_filter=True)
print("🌍 Idioma detectado: '%s' (probabilidad %.2f)" % (info.language, info.language_probability))

transcription = []
print("📝 Iniciando transcripción...")
for segment in tqdm(segments, desc="⏳ Transcribiendo", unit="segmento"):
    transcription.append("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# ▌GUARDAR TRANSCRIPCIÓN
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(transcription))

# ▌TIEMPO FINAL
elapsed = time.time() - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

print(f"\n✅ Transcripción guardada en:\n{output_path}")
print(f"⏱️ Tiempo total de transcripción: {minutes} minutos y {seconds} segundos")
