import os
import re
import threading
import time
from typing import Callable, Dict, Tuple

import psutil

try:
    import torch
except ImportError:
    torch = None


def compute_type_token_ratio(text: str) -> float:
    """Calculates lexical diversity: unique words / total words."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


class PeakMemorySampler:
    """Samples process RSS memory in a background thread to capture peak RAM usage."""

    def __init__(self, interval_sec: float = 0.01):
        self.interval = interval_sec
        self.process = psutil.Process(os.getpid())
        self.stop_signal = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_bytes = self.process.memory_info().rss

    def _sample(self):
        while not self.stop_signal.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval)

    def start(self):
        self.thread.start()

    def stop(self) -> int:
        self.stop_signal.set()
        self.thread.join(timeout=0.2)
        final_rss = self.process.memory_info().rss
        return max(self.peak_bytes, final_rss)


def profile_inference(
    inference_func: Callable[[], Tuple[str, int]],
    min_quality_tokens: int = 5,
) -> Dict[str, float | str | bool]:
    has_cuda = torch is not None and torch.cuda.is_available()

    if has_cuda:
        torch.cuda.reset_peak_memory_stats()

    sampler = PeakMemorySampler()
    sampler.start()

    start_time = time.perf_counter()
    output_text, token_count = inference_func()
    end_time = time.perf_counter()

    peak_ram_bytes = sampler.stop()

    peak_vram_bytes = 0
    if has_cuda:
        peak_vram_bytes = torch.cuda.max_memory_allocated()

    latency = max(end_time - start_time, 1e-6)
    throughput = round(token_count / latency, 2)
    ttr = compute_type_token_ratio(output_text)
    is_degenerate = token_count < min_quality_tokens or (token_count > 10 and ttr < 0.15)

    return {
        "output_text": output_text,
        "token_count": token_count,
        "latency_sec": round(latency, 4),
        "throughput_tps": throughput,
        "peak_ram_mb": round(peak_ram_bytes / (1024 * 1024), 2),
        "peak_vram_mb": round(peak_vram_bytes / (1024 * 1024), 2) if has_cuda else 0.0,
        "quality_ttr": ttr,
        "is_degenerate": is_degenerate,
    }
