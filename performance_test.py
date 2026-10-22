"""
Performance test for the Smart Recycling Detector.

Measures: model load time, memory, preprocessing time, ML inference time,
total detect_recycling() time, and CPU usage.
"""
import base64
import contextlib
import io
import os
import platform
import statistics
import sys
import time

import numpy as np
import psutil
from PIL import Image

WARMUP_RUNS = 5   # first predictions are slow while TensorFlow warms up
TEST_RUNS = 50

proc = psutil.Process(os.getpid())


def rss_mb():
    return proc.memory_info().rss / 1024 ** 2


def stats(values_ms):
    s = sorted(values_ms)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return {
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p95": p95,
        "min": s[0],
        "max": s[-1],
    }


image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
if not os.path.exists(image_path):
    sys.exit(f"Image not found: {image_path}. Put a photo of an object in this folder.")

# ---- model load time and memory ------------------------------------------
mem_start = rss_mb()
t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    import detector  # loads TensorFlow and the model
load_time = time.perf_counter() - t0
mem_loaded = rss_mb()

if detector.model is None:
    sys.exit("Real model NOT loaded (mock mode). Fix the model path first; "
             "mock mode sleeps 1.2 s and its timings are meaningless.")

# ---- test image as the browser would send it (base64 data URL) -----------
with open(image_path, "rb") as f:
    raw = f.read()
b64 = "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def preprocess(data_url):
    """Same steps as detector.detect_recycling, before model.predict."""
    payload = data_url.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(payload)))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = detector.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)


def quiet_detect(data_url):
    with contextlib.redirect_stdout(io.StringIO()):
        return detector.detect_recycling(data_url)


# ---- warm-up --------------------------------------------------------------
x = preprocess(b64)
first_start = time.perf_counter()
detector.model.predict(x, verbose=0)
first_inference_ms = (time.perf_counter() - first_start) * 1000
for _ in range(WARMUP_RUNS):
    quiet_detect(b64)

# ---- idle baseline --------------------------------------------------------
proc.cpu_percent(interval=None)
time.sleep(3)
idle_cpu = proc.cpu_percent(interval=None)
idle_mem = rss_mb()

# ---- timed runs -----------------------------------------------------------
pre_ms, inf_ms, total_ms = [], [], []
peak_mem = idle_mem
proc.cpu_percent(interval=None)
wall_start = time.perf_counter()
for _ in range(TEST_RUNS):
    t = time.perf_counter()
    x = preprocess(b64)
    pre_ms.append((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    detector.model.predict(x, verbose=0)
    inf_ms.append((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    quiet_detect(b64)
    total_ms.append((time.perf_counter() - t) * 1000)

    peak_mem = max(peak_mem, rss_mb())
wall = time.perf_counter() - wall_start
busy_cpu = proc.cpu_percent(interval=None)  # 100% = one core fully used
cores = psutil.cpu_count(logical=True)

# ---- report ---------------------------------------------------------------
print("\n=== Environment ===")
print(f"OS: {platform.platform()}")
print(f"CPU: {platform.processor()} ({psutil.cpu_count(logical=False)} cores / {cores} threads)")
print(f"RAM: {psutil.virtual_memory().total / 1024 ** 3:.1f} GB")
print(f"Python: {platform.python_version()}")
try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
except Exception:
    pass
print(f"Test image: {image_path} ({len(raw) / 1024:.0f} KB), runs: {TEST_RUNS} (after {WARMUP_RUNS} warm-up)")

print("\n=== Timing (milliseconds) ===")
print("| Stage | Mean | Median | 95th pct | Min | Max |")
print("|---|---|---|---|---|---|")
for name, vals in [("Preprocessing (decode + resize)", pre_ms),
                   ("ML inference (model.predict)", inf_ms),
                   ("Total detect_recycling()", total_ms)]:
    s = stats(vals)
    print(f"| {name} | {s['mean']:.1f} | {s['median']:.1f} | {s['p95']:.1f} | {s['min']:.1f} | {s['max']:.1f} |")
print(f"\nFirst inference after load (cold): {first_inference_ms:.0f} ms")
print(f"Model load time (import detector): {load_time:.1f} s")

print("\n=== Memory (this Python process, MB) ===")
print(f"Before importing detector: {mem_start:.0f}")
print(f"After model load:          {mem_loaded:.0f}")
print(f"Idle:                      {idle_mem:.0f}")
print(f"Peak during {TEST_RUNS} runs:     {peak_mem:.0f}")

print("\n=== CPU ===")
print(f"Idle: {idle_cpu:.1f}%")
print(f"During runs: {busy_cpu:.0f}% of one core "
      f"(about {busy_cpu / cores:.0f}% of the whole CPU with {cores} threads)")
print(f"Throughput: {1000 / statistics.mean(total_ms):.1f} detections per second")
