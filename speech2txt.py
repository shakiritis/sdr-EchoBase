import time, threading, queue, sys, signal
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from math import gcd

# ====== Config ======
MODEL_SIZE = "base"         # choose from "base", "small", "medium", "large", "tiny"
DEVICE = "cpu"              # "cpu" or "auto"
COMPUTE_TYPE = "int8"       # "float16" if running on GPU

CHANNELS = 1
TARGET_SR = 16000 # sample rate
TRY_SR = [48000, 44100, 32000, 22050, 16000] # sample rates to try

# Audio buffering / batching
LATENCY_SEC = 0.25          # numeric seconds; increase to 0.35–0.50 if needed
BLOCKSIZE = 0               # 0 = let host choose buffer size
TRANSCRIBE_CHUNK_SECONDS = 5.0
ROLLING_CONTEXT_SECONDS  = 30.0

# Queues & flags
audio_q  = queue.Queue(maxsize=64)   # bounded; if full we drop oldest
status_q = queue.Queue(maxsize=16)
stop_flag = False

def handle_sigint(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, handle_sigint)

# ---------- device/samplerate ----------
def pick_device_and_sr():
    print("\n=== Devices (input) ===")
    hostapis = sd.query_hostapis()
    inputs = []
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            host = hostapis[d['hostapi']]['name']
            print(f"{i}: {d['name']}  (hostapi={host})")
            inputs.append(i)

    dev = sd.default.device[0]
    if dev is None or sd.query_devices(dev)['max_input_channels'] == 0:
        dev = inputs[0]

    for sr in TRY_SR:
        try:
            sd.check_input_settings(device=dev, samplerate=sr, channels=CHANNELS)
            return dev, sr
        except Exception:
            continue

    sr = int(sd.query_devices(dev)['default_samplerate'])
    sd.check_input_settings(device=dev, samplerate=sr, channels=CHANNELS)
    return dev, sr

# ---------- audio I/O ----------
def audio_callback(indata, frames, time_info, status):
    # Never print here; it can cause or amplify overflow.
    if status:
        try:
            if status.input_overflow:
                status_q.put_nowait("input overflow")
            else:
                status_q.put_nowait(str(status))
        except queue.Full:
            pass

    try:
        if audio_q.full():
            # drop oldest to keep up (prefer minor loss over device overflow)
            audio_q.get_nowait()
        audio_q.put_nowait(indata.copy())
    except queue.Full:
        pass

def start_input_stream(device_index, sr_device):
    return sd.InputStream(
        samplerate=sr_device,
        channels=CHANNELS,
        dtype="float32",
        device=device_index,
        latency=LATENCY_SEC,     # numeric seconds
        blocksize=BLOCKSIZE,     # let host decide
        callback=audio_callback,
    )

# ---------- helpers ----------
def resample_float32_mono(x_f32, sr_from, sr_to):
    if sr_from == sr_to:
        return x_f32.astype(np.float32).flatten()
    g = gcd(sr_from, sr_to)
    up, down = sr_to // g, sr_from // g
    y = resample_poly(x_f32.flatten(), up, down)
    return np.clip(y, -1.0, 1.0).astype(np.float32)

def stdin_quitter():
    global stop_flag
    for line in sys.stdin:
        if line.strip().lower() == "q":
            stop_flag = True
            break

def status_reporter():
    last = 0.0
    while not stop_flag:
        now = time.time()
        if now - last >= 1.0:
            msgs = []
            while True:
                try:
                    msgs.append(status_q.get_nowait())
                except queue.Empty:
                    break
            if msgs:
                counts = {}
                for m in msgs:
                    counts[m] = counts.get(m, 0) + 1
                summary = ", ".join(f"{k} x{v}" for k, v in counts.items())
                print(f"[Audio status] {summary}", file=sys.stderr, flush=True)
            last = now
        time.sleep(0.05)

# ---------- transcription worker ----------
def transcriber_worker(sr_device, model: WhisperModel, fout):
    need_16k = int(TARGET_SR * TRANSCRIBE_CHUNK_SECONDS)
    roll_max = int(TARGET_SR * ROLLING_CONTEXT_SECONDS)
    rolling = np.zeros(0, dtype=np.float32)

    stage_dev = np.zeros(0, dtype=np.float32)  # device‑SR staging

    while not stop_flag:
        try:
            blk = audio_q.get(timeout=0.25)  # float32 at device SR
        except queue.Empty:
            continue

        stage_dev = np.concatenate([stage_dev, blk.flatten()])

        # approx how many device samples correspond to need_16k target samples
        dev_needed = int(need_16k * (sr_device / TARGET_SR))
        if stage_dev.size >= dev_needed:
            take = stage_dev[:dev_needed]
            stage_dev = stage_dev[dev_needed:]

            chunk16 = resample_float32_mono(take, sr_device, TARGET_SR)

            rolling = np.concatenate([rolling, chunk16])
            if rolling.size > roll_max:
                rolling = rolling[-roll_max:]

            segments, _ = model.transcribe(
                chunk16,
                vad_filter=True,
                beam_size=1,
                temperature=0.0,
                word_timestamps=False,
                language=None,
            )

            pieces = [seg.text.strip() for seg in segments if seg.text.strip()]
            if pieces:
                line = " ".join(pieces)
                print(line, flush=True)
                fout.write(line + "\n"); fout.flush()

# ---------- main ----------
def main():
    global stop_flag

    dev, sr_device = pick_device_and_sr()
    info = sd.query_devices(dev)
    host = sd.query_hostapis()[info['hostapi']]['name']
    print(f"Input device: #{dev} '{info['name']}' via {host}")
    print(f"Device SR: {sr_device} Hz  → target {TARGET_SR} Hz")
    print(f"Opening stream with latency={LATENCY_SEC}s, blocksize={BLOCKSIZE}")

    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Speak — press Ctrl+C to stop.")

    # start audio stream (context manager keeps it open)
    with start_input_stream(dev, sr_device):
        # side threads
        t_quit  = threading.Thread(target=stdin_quitter, daemon=True); t_quit.start()
        t_stat  = threading.Thread(target=status_reporter, daemon=True); t_stat.start()

        with open("transcript.txt", "w", encoding="utf-8") as fout:
            t_worker = threading.Thread(target=transcriber_worker,
                                        args=(sr_device, model, fout),
                                        daemon=True)
            t_worker.start()

            try:
                while not stop_flag:
                    time.sleep(0.2)
            finally:
                stop_flag = True
                time.sleep(0.5)

    print("\nStopped. Transcript saved to transcript.txt")

if __name__ == "__main__":
    main()

