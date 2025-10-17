import multiprocessing
import os
try:
	import torch  # Optional: to decide preload strategy for GPU
except Exception:
	torch = None

# Bind to PORT env var if provided (Render/Heroku/Azure), else default 8080
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# Workers: 2-4 x CPU cores is a common guideline; start modestly
workers = int(os.environ.get("WEB_CONCURRENCY", max(2, multiprocessing.cpu_count())))

# Threads can help with I/O; Flask + PyTorch is mostly CPU-bound when doing inference
threads = int(os.environ.get("THREADS", 2))

# Preload the application code before forking workers to reduce memory footprint (CPU only)
# For CUDA/MPS (GPU) backends, preloading can cause context issues; disable if GPU is present
gpu_available = False
if torch is not None:
	try:
		gpu_available = bool(torch.cuda.is_available() or getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
	except Exception:
		gpu_available = False

# Allow override via PRELOAD_APP env var ("1"/"0")
_preload_env = os.environ.get("PRELOAD_APP")
preload_app = (not gpu_available) if _preload_env is None else (_preload_env == "1")

# Increase timeout for potential model warmup / first inference
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 120))

# Access log to stdout for container platforms
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# Graceful settings
graceful_timeout = 30
keepalive = 5
