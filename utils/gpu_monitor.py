"""
GPU monitoring utilities for tracking GPU utilization, memory, power, and temperature.

This module provides background GPU telemetry sampling and per-iteration CUDA memory
tracking for experiment monitoring and resource optimization.

NOTE: Currently supports single-GPU monitoring only. Multi-GPU support needs to be
extended in the future to aggregate metrics across multiple devices.
"""

import os
import csv
import time
import threading
import torch
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Import tqdm for progress bar compatible printing
try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# Optional NVML import
try:
    import pynvml  # provided by nvidia-ml-py3
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

# Optional GPUtil fallback
try:
    import GPUtil  # type: ignore
    _GPUTIL_AVAILABLE = True
except Exception:
    _GPUTIL_AVAILABLE = False


class GpuMonitor:
    """
    Background GPU telemetry sampler using NVML (preferred) or GPUtil fallback.
    
    Periodically samples GPU utilization, memory use, power, and temperature in a
    background thread. Writes a CSV timeline on stop() and provides aggregate
    statistics.
    
    NOTE: Currently monitors a single GPU device. Multi-GPU support needs to be
    extended to aggregate metrics across multiple devices.
    
    Usage:
        monitor = GpuMonitor(device_index=0, out_csv="gpu_telemetry.csv")
        monitor.start()
        # ... run experiment ...
        monitor.stop()
        summary = monitor.summary()
    """

    def __init__(self, device_index: int, out_csv: str, interval_s: float = 0.5):
        """
        Initialize GPU monitor.
        
        Args:
            device_index: GPU device index to monitor
            out_csv: Path to output CSV file for telemetry data
            interval_s: Sampling interval in seconds (default: 0.5)
        """
        self.device_index = int(device_index)
        self.out_csv = out_csv
        self.interval_s = float(interval_s)
        self._rows = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self._use_nvml = _NVML_AVAILABLE
        self._handle = None
        self._log_callback = None
        self._log_interval_s = 30.0  # Default: log every 30 seconds
        self._log_thread: Optional[threading.Thread] = None
        self._last_log_time = 0.0
        self._last_log_index = 0

    def start(self):
        """
        Start background sampling thread.
        
        No-op if neither NVML nor GPUtil is available. Creates output directory
        if needed.
        """
        if self._started:
            return
        
        # Create output directory if needed
        out_dir = os.path.dirname(self.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        # Initialize NVML if available
        if self._use_nvml:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            except Exception:
                self._use_nvml = False
        
        # Check if any monitoring backend is available
        if not self._use_nvml and not _GPUTIL_AVAILABLE:
            # Nothing to sample; leave disabled
            return
        
        # Start background thread
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started = True

    def _run(self):
        """
        Background sampling loop.
        
        Continuously samples GPU metrics at the specified interval until stop()
        is called. Swallows sampling errors to ensure continuous operation.
        """
        while not self._stop.is_set():
            timestamp = datetime.utcnow().isoformat()
            try:
                if self._use_nvml:
                    # Use NVML for detailed metrics
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    power_w = None
                    temp_c = None
                    
                    # Power and temperature may not be available on all GPUs
                    try:
                        power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                    except Exception:
                        pass
                    try:
                        temp_c = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        pass

                    row = [
                        timestamp,
                        util.gpu,
                        util.memory,
                        int(mem.used // (1024 * 1024)),  # Convert to MiB
                        int(mem.total // (1024 * 1024)),
                        power_w if power_w is not None else "",
                        temp_c if temp_c is not None else "",
                    ]
                else:
                    # GPUtil fallback
                    gpus = GPUtil.getGPUs()
                    gpu = next((g for g in gpus if g.id == self.device_index), None)
                    if gpu is None:
                        time.sleep(self.interval_s)
                        continue
                    row = [
                        timestamp,
                        int(gpu.load * 100),
                        int((gpu.memoryUtil or 0) * 100),
                        int(gpu.memoryUsed),
                        int(gpu.memoryTotal),
                        "",
                        gpu.temperature if hasattr(gpu, 'temperature') else "",
                    ]
                self._rows.append(row)
            except Exception:
                # Swallow sampling errors; continue monitoring
                pass
            time.sleep(self.interval_s)

    def stop(self):
        """
        Stop sampling and write CSV to disk.
        
        Joins the background threads and writes all collected samples to CSV.
        Cleans up NVML if it was used.
        """
        if not self._started:
            return
        
        # Signal threads to stop
        self._stop.set()
        
        # Wait for sampling thread to finish
        if self._thread is not None:
            self._thread.join()
        
        # Wait for logging thread to finish
        if self._log_thread is not None:
            self._log_thread.join()
        
        # Write CSV file
        try:
            with open(self.out_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp",
                    "util_gpu_pct",
                    "util_mem_pct",
                    "mem_used_mib",
                    "mem_total_mib",
                    "power_w",
                    "temp_c"
                ])
                w.writerows(self._rows)
        finally:
            # Clean up NVML
            if self._use_nvml:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    def summary(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics from collected samples.
        
        Returns:
            Dictionary with aggregate metrics (avg, p95, max) or empty dict
            if no samples collected.
        """
        if not self._rows:
            return {}
        
        # Extract columns: timestamp, util_gpu_pct, util_mem_pct, mem_used_mib, mem_total_mib, power_w, temp_c
        util = [int(r[1]) for r in self._rows if isinstance(r[1], int)]
        mem_used = [int(r[3]) for r in self._rows if isinstance(r[3], int)]
        mem_total = max([int(r[4]) for r in self._rows if isinstance(r[4], int)]) if any(isinstance(r[4], int) for r in self._rows) else None
        power = [float(r[5]) for r in self._rows if isinstance(r[5], (int, float))]
        temp = [int(r[6]) for r in self._rows if isinstance(r[6], int)]

        def _avg(xs):
            """Compute average of a list."""
            return sum(xs) / len(xs) if xs else None

        def _p95(xs):
            """Compute 95th percentile of a list."""
            if not xs:
                return None
            ys = sorted(xs)
            idx = int(0.95 * (len(ys) - 1))
            return ys[idx]

        return {
            'avg_util_gpu_pct': _avg(util),
            'p95_util_gpu_pct': _p95(util),
            'avg_mem_used_mib': _avg(mem_used),
            'p95_mem_used_mib': _p95(mem_used),
            'max_mem_used_mib': max(mem_used) if mem_used else None,
            'mem_total_mib': mem_total,
            'avg_power_w': _avg(power),
            'max_temp_c': max(temp) if temp else None,
            'num_samples': len(self._rows),
        }

    def sample_count(self) -> int:
        """Return number of collected samples so far."""
        return len(self._rows)
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Get the most recent GPU metrics sample.
        
        Returns:
            Dictionary with latest metrics or empty dict if no samples available
        """
        if not self._rows:
            return {}
        
        # Get the most recent row
        latest = self._rows[-1]
        # Columns: timestamp, util_gpu_pct, util_mem_pct, mem_used_mib, mem_total_mib, power_w, temp_c
        return {
            'timestamp': latest[0],
            'util_gpu_pct': latest[1] if isinstance(latest[1], int) else None,
            'util_mem_pct': latest[2] if isinstance(latest[2], int) else None,
            'mem_used_mib': latest[3] if isinstance(latest[3], int) else None,
            'mem_total_mib': latest[4] if isinstance(latest[4], int) else None,
            'power_w': latest[5] if isinstance(latest[5], (int, float)) else None,
            'temp_c': latest[6] if isinstance(latest[6], int) else None,
        }
    
    def enable_periodic_logging(self, log_callback, log_interval_s: float = 30.0):
        """
        Enable periodic logging of GPU metrics.
        
        Starts a background thread that periodically calls the log_callback with
        incremental GPU statistics. Useful for real-time monitoring and wandb logging.
        
        Args:
            log_callback: Callback function that takes a dict of metrics and logs them
            log_interval_s: Interval between log calls in seconds (default: 30.0)
        """
        if not self._started:
            raise RuntimeError("Monitor must be started before enabling periodic logging")
        
        self._log_callback = log_callback
        self._log_interval_s = float(log_interval_s)
        self._last_log_time = time.time()
        self._last_log_index = 0
        
        # Start logging thread
        self._log_thread = threading.Thread(target=self._log_loop, daemon=True)
        self._log_thread.start()
    
    def _log_loop(self):
        """
        Background thread that periodically logs GPU metrics.
        
        Uses summarize_since() to get incremental statistics and calls the
        logging callback with the metrics.
        """
        while not self._stop.is_set():
            time.sleep(self._log_interval_s)
            
            if self._stop.is_set():
                break
            
            # Get incremental summary since last log
            summary = self.summarize_since(self._last_log_index)
            
            if summary.get('num_samples', 0) > 0 and self._log_callback:
                # Prepare metrics for logging
                metrics = {
                    'gpu_avg_util_pct': summary.get('avg_util_gpu_pct'),
                    'gpu_avg_mem_used_mib': summary.get('avg_mem_used_mib'),
                    'gpu_max_mem_used_mib': summary.get('max_mem_used_mib'),
                    'gpu_mem_total_mib': summary.get('mem_total_mib'),
                    'gpu_avg_power_w': summary.get('avg_power_w'),
                    'gpu_max_temp_c': summary.get('max_temp_c'),
                }
                # Filter out None values
                metrics = {k: v for k, v in metrics.items() if v is not None}
                
                if metrics:
                    try:
                        self._log_callback(metrics)
                    except Exception:
                        # Swallow logging errors to avoid disrupting monitoring
                        pass
            
            # Update cursor for next iteration
            self._last_log_index = summary.get('end_index', self._last_log_index)
            self._last_log_time = time.time()

    def summarize_since(self, start_index: int) -> Dict[str, Any]:
        """
        Compute aggregate statistics from samples collected since start_index.
        
        Useful for incremental monitoring during long-running experiments.
        Returns an empty dict if no new samples are available.
        
        Args:
            start_index: Starting index for samples to include
            
        Returns:
            Dictionary with aggregate metrics and end_index for cursor advancement
        """
        n = len(self._rows)
        if start_index is None or start_index < 0 or start_index >= n:
            # Nothing new or invalid index
            return {
                'end_index': n,
                'num_samples': 0,
            }
        
        rows = self._rows[start_index:]

        def _avg(xs):
            """Compute average of a list."""
            return sum(xs) / len(xs) if xs else None

        util = [int(r[1]) for r in rows if isinstance(r[1], int)]
        mem_used = [int(r[3]) for r in rows if isinstance(r[3], int)]
        mem_total = max([int(r[4]) for r in rows if isinstance(r[4], int)]) if any(isinstance(r[4], int) for r in rows) else None
        power = [float(r[5]) for r in rows if isinstance(r[5], (int, float))]
        temp = [int(r[6]) for r in rows if isinstance(r[6], int)]

        return {
            'end_index': n,
            'num_samples': len(rows),
            'avg_util_gpu_pct': _avg(util),
            'avg_mem_used_mib': _avg(mem_used),
            'max_mem_used_mib': max(mem_used) if mem_used else None,
            'mem_total_mib': mem_total,
            'avg_power_w': _avg(power),
            'max_temp_c': max(temp) if temp else None,
        }


class StepMonitor:
    """
    Per-iteration timing and CUDA memory snapshots.
    
    Tracks CUDA memory usage at named checkpoints during training iterations.
    Useful for profiling memory usage patterns and identifying bottlenecks.
    
    Usage:
        sm = StepMonitor(enable=torch.cuda.is_available())
        sm.epoch_start()
        sm.mark('pre_forward')
        ... forward pass ...
        sm.mark('post_forward')
        ... backward pass ...
        sm.mark('post_backward')
        ... optimizer.step() ...
        sm.mark('post_optim')
        info = sm.finalize_step()
    """

    def __init__(self, enable: bool = True):
        """
        Initialize step monitor.
        
        Args:
            enable: Whether to enable monitoring (default: True)
        """
        self.enable = bool(enable)
        self._t0 = None
        self._marks: Dict[str, Dict[str, Any]] = {}
        self._epoch_peak_alloc = 0
        self._epoch_peak_reserved = 0

    def epoch_start(self):
        """
        Reset peak memory stats at the start of an epoch.
        
        Resets PyTorch CUDA peak memory statistics if CUDA is available.
        """
        if not self.enable:
            return
        try:
            if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
                import torch  # local import to avoid overhead if disabled
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self._epoch_peak_alloc = 0
        self._epoch_peak_reserved = 0

    def _cuda_mem(self):
        """
        Get current CUDA memory statistics.
        
        Returns:
            Tuple of (allocated, reserved, max_allocated, max_reserved) in bytes
        """
        try:
            import torch
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated()
            reserv = torch.cuda.memory_reserved()
            max_alloc = torch.cuda.max_memory_allocated()
            max_reserv = torch.cuda.max_memory_reserved()
            return alloc, reserv, max_alloc, max_reserv
        except Exception:
            return 0, 0, 0, 0

    def mark(self, name: str):
        """
        Record a memory snapshot at a named checkpoint.
        
        Args:
            name: Name of the checkpoint (e.g., 'pre_forward', 'post_backward')
        """
        if not self.enable:
            return
        now = time.perf_counter()
        alloc, reserv, max_alloc, max_reserv = self._cuda_mem()
        self._epoch_peak_alloc = max(self._epoch_peak_alloc, max_alloc)
        self._epoch_peak_reserved = max(self._epoch_peak_reserved, max_reserv)
        self._marks[name] = {
            'ts': now,
            'alloc': alloc,
            'reserved': reserv,
            'peak_alloc': max_alloc,
            'peak_reserved': max_reserv,
        }

    def finalize_step(self) -> Dict[str, Any]:
        """
        Finalize current step and return all marks.
        
        Clears internal marks dictionary after returning data.
        
        Returns:
            Dictionary of checkpoint names to memory statistics
        """
        if not self.enable:
            return {}
        marks = self._marks
        self._marks = {}
        return marks

    @staticmethod
    def bytes_to_mib(x: int) -> float:
        """Convert bytes to MiB."""
        return float(x) / (1024.0 * 1024.0)

    def epoch_peaks_mib(self) -> Dict[str, float]:
        """
        Get epoch peak memory usage in MiB.
        
        Returns:
            Dictionary with peak_alloc_mib and peak_reserved_mib
        """
        return {
            'peak_alloc_mib': self.bytes_to_mib(self._epoch_peak_alloc),
            'peak_reserved_mib': self.bytes_to_mib(self._epoch_peak_reserved),
        }


def _format_gpu_metrics(metrics: Dict[str, Any], window_seconds: Optional[float]) -> list:
    """
    Format GPU metrics into two lines for better readability.
    
    Args:
        metrics: Dictionary of GPU metrics
        window_seconds: Time window for the metrics (for display). If None, indicates final summary.
    
    Returns:
        List of two formatted strings (one per line)
    """
    # Convert MiB to GB for memory metrics
    def mib_to_gb(mib: Optional[float]) -> Optional[float]:
        """Convert MiB to GB."""
        if mib is None:
            return None
        return mib / 1024.0
    
    # Extract and format metrics
    avg_util = metrics.get('gpu_avg_util_pct')
    avg_mem = mib_to_gb(metrics.get('gpu_avg_mem_used_mib'))
    max_mem = mib_to_gb(metrics.get('gpu_max_mem_used_mib'))
    mem_total = mib_to_gb(metrics.get('gpu_mem_total_mib'))
    avg_power = metrics.get('gpu_avg_power_w')
    max_temp = metrics.get('gpu_max_temp_c')
    
    # Build first line: Utilization and Memory
    line1_parts = []
    if avg_util is not None:
        line1_parts.append(f"Avg Util: {avg_util:.1f}%")
    if avg_mem is not None:
        line1_parts.append(f"Avg Mem: {avg_mem:.2f} GB")
    if max_mem is not None:
        line1_parts.append(f"Max Mem: {max_mem:.2f} GB")
    if mem_total is not None:
        line1_parts.append(f"Mem Total: {mem_total:.2f} GB")
    
    # Build second line: Power and Temperature
    line2_parts = []
    if avg_power is not None:
        line2_parts.append(f"Avg Power: {avg_power:.0f}W")
    if max_temp is not None:
        line2_parts.append(f"Max Temp: {max_temp}°C")
    
    # Format lines with spacing
    line1 = "   ".join(line1_parts) if line1_parts else ""
    line2 = "   ".join(line2_parts) if line2_parts else ""
    
    # Create header with window indicator
    if window_seconds is not None:
        header = f"[GPU Monitor (last {int(window_seconds)}s)]"
    else:
        header = "[GPU Monitor (final summary - entire run)]"
    
    return [f"{header} {line1}", f"{' ' * len(header)} {line2}"]


# Track previous GPU monitor line count for overwriting
_previous_gpu_monitor_lines = 0


def _print_gpu_metrics(lines: list):
    """
    Print GPU metrics in a tqdm-compatible way with line overwriting.
    
    Uses tqdm.write() to avoid interfering with progress bars, and overwrites
    the previous GPU monitor output to keep the display clean.
    
    Args:
        lines: List of formatted strings to print (typically 2 lines)
    """
    global _previous_gpu_monitor_lines
    
    if not lines:
        return
    
    # Use tqdm.write() if available, otherwise fall back to regular print
    if _TQDM_AVAILABLE:
        # Move cursor up to overwrite previous GPU monitor lines
        if _previous_gpu_monitor_lines > 0:
            # ANSI escape: move up N lines
            tqdm.write(f"\033[{_previous_gpu_monitor_lines}A", end="")
        
        # Print new lines (tqdm.write() ensures they appear below progress bar)
        for line in lines:
            # Clear line and print new content
            tqdm.write(f"\033[K{line}")
        
        # Update line count for next iteration
        _previous_gpu_monitor_lines = len(lines)
    else:
        # Fallback for when tqdm is not available
        # Move cursor up to overwrite previous lines
        if _previous_gpu_monitor_lines > 0:
            print(f"\033[{_previous_gpu_monitor_lines}A", end="")
        
        for line in lines:
            # Clear line and print new content
            print(f"\033[K{line}")
        
        _previous_gpu_monitor_lines = len(lines)


@contextmanager
def gpu_monitoring_context(args, exp_manager, log_interval_s: float = 30.0):
    """
    Context manager for GPU monitoring during experiment execution.
    
    Handles GPU monitor initialization, periodic logging setup, and cleanup.
    Automatically logs metrics to ExperimentManager (wandb + local files) and console.
    
    Args:
        args: Arguments object with GPU configuration (use_gpu, gpu, use_multi_gpu, device_ids)
        exp_manager: ExperimentManager instance for logging metrics
        log_interval_s: Interval between periodic log updates in seconds (default: 30.0)
    
    Yields:
        GpuMonitor instance if GPU monitoring is active, None otherwise
    
    Example:
        with gpu_monitoring_context(args, exp_manager) as gpu_monitor:
            # Run experiment
            exp.train()
        # GPU monitoring automatically stopped and final metrics logged
    """
    gpu_monitor = None
    
    # Initialize GPU monitor if GPU is available
    if args.use_gpu and torch.cuda.is_available():
        # Determine GPU device index
        if hasattr(args, 'use_multi_gpu') and args.use_multi_gpu and hasattr(args, 'device_ids'):
            gpu_id = args.device_ids[0]
        else:
            gpu_id = args.gpu
        
        # Set up GPU monitor
        gpu_csv_path = str(exp_manager.get_log_dir() / "gpu_telemetry.csv")
        gpu_monitor = GpuMonitor(device_index=gpu_id, out_csv=gpu_csv_path)
        gpu_monitor.start()
        
        # Enable periodic logging to wandb and log file
        # Note: log_interval_s (30 seconds) is currently hardcoded but can be made configurable later
        def log_gpu_metrics(metrics: Dict[str, Any]):
            """Callback to log GPU metrics periodically."""
            # Log to ExperimentManager (which logs to wandb and local files)
            exp_manager.log_metrics(metrics)
            
            # Format and print GPU metrics in a tqdm-compatible way
            # Window indicator shows the time window for these metrics (last N seconds)
            formatted_lines = _format_gpu_metrics(metrics, log_interval_s)
            _print_gpu_metrics(formatted_lines)
        
        gpu_monitor.enable_periodic_logging(log_gpu_metrics, log_interval_s=log_interval_s)
        
        # Register GPU monitor with ExperimentManager so end_experiment() can access it
        exp_manager.gpu_monitor = gpu_monitor
    
    try:
        yield gpu_monitor
    finally:
        # Reset GPU monitor line counter for clean exit
        global _previous_gpu_monitor_lines
        _previous_gpu_monitor_lines = 0
        # Stop GPU monitoring and display final summary
        # Note: GPU metrics were already logged to wandb in end_experiment() before run finished
        if gpu_monitor:
            gpu_monitor.stop()
            summary = gpu_monitor.summary()
            if summary:
                gpu_metrics = {
                    'gpu_avg_util_pct': summary.get('avg_util_gpu_pct'),
                    'gpu_p95_util_pct': summary.get('p95_util_gpu_pct'),
                    'gpu_avg_mem_used_mib': summary.get('avg_mem_used_mib'),
                    'gpu_max_mem_used_mib': summary.get('max_mem_used_mib'),
                    'gpu_mem_total_mib': summary.get('mem_total_mib'),
                    'gpu_avg_power_w': summary.get('avg_power_w'),
                    'gpu_max_temp_c': summary.get('max_temp_c'),
                    'gpu_num_samples': summary.get('num_samples'),
                }
                # Filter out None values
                gpu_metrics = {k: v for k, v in gpu_metrics.items() if v is not None}
                if gpu_metrics:
                    # Log to local files (wandb was already updated in end_experiment())
                    exp_manager.log_metrics(gpu_metrics)
                    
                    # Format and display final summary (over entire run, not a window)
                    final_lines = _format_gpu_metrics(gpu_metrics, window_seconds=None)
                    _print_gpu_metrics(final_lines)
            
            # Clear GPU monitor reference
            exp_manager.gpu_monitor = None
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

