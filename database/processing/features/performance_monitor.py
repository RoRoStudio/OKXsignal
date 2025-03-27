# database/processing/features/performance_monitor.py
#!/usr/bin/env python3
"""
Performance Monitoring for feature computation
"""

import os
import logging
import threading
import time
import json
import numpy as np
from datetime import datetime
import psutil

# Try to import GPU monitoring library
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.debug("pynvml not available. GPU monitoring will be disabled.")

class ResourceMonitor:
    """Monitor system resource utilization during feature computation"""
    
    def __init__(self):
        """Initialize the resource monitor"""
        self.cpu_samples = []
        self.ram_samples = []
        self.gpu_samples = []
        self.sample_timestamps = []
        self.initialized = False
        
        # Try to initialize GPU monitoring
        self.gpu_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.num_gpus = pynvml.nvmlDeviceGetCount()
                self.gpu_available = self.num_gpus > 0
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]
                gpu_names = []
                for handle in self.gpu_handles:
                    try:
                        gpu_names.append(pynvml.nvmlDeviceGetName(handle).decode('utf-8'))
                    except:
                        gpu_names.append("Unknown GPU")
                self.gpu_names = gpu_names
                logging.info(f"GPU monitoring initialized for {self.num_gpus} GPUs: {', '.join(self.gpu_names)}")
            except Exception as e:
                logging.warning(f"GPU monitoring initialization failed: {e}")
                self.gpu_available = False
        
        # System information
        self.total_ram = psutil.virtual_memory().total
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cpu_count = psutil.cpu_count(logical=False)
        self.initialized = True
        
        logging.info(f"Resource monitor initialized: CPU cores: {self.physical_cpu_count} physical / {self.cpu_count} logical, "
                    f"RAM: {self.total_ram / (1024**3):.2f} GB")
        
    def take_sample(self):
        """Take a sample of current resource utilization"""
        if not self.initialized:
            return
            
        # Record timestamp
        self.sample_timestamps.append(time.time())
        
        # CPU utilization (percentage)
        self.cpu_samples.append(psutil.cpu_percent(interval=0.1, percpu=True))
        
        # RAM utilization
        ram = psutil.virtual_memory()
        self.ram_samples.append({
            'percent': ram.percent,
            'used': ram.used,
            'available': ram.available
        })
        
        # GPU utilization if available
        if self.gpu_available:
            gpu_stats = []
            for i, handle in enumerate(self.gpu_handles):
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    gpu_stats.append({
                        'gpu_id': i,
                        'name': self.gpu_names[i],
                        'utilization': utilization.gpu,
                        'memory_used': memory.used,
                        'memory_total': memory.total,
                        'memory_percent': (memory.used / memory.total) * 100,
                        'temperature': temperature
                    })
                except Exception as e:
                    gpu_stats.append({
                        'gpu_id': i,
                        'name': self.gpu_names[i] if i < len(self.gpu_names) else "Unknown",
                        'error': str(e)
                    })
            
            self.gpu_samples.append(gpu_stats)
            
    def get_summary(self):
        """Get a summary of resource utilization"""
        if not self.initialized or not self.cpu_samples:
            return {
                'cpu': {'available': False},
                'ram': {'available': False},
                'gpu': {'available': False}
            }
            
        # Calculate statistics
        avg_cpu = np.mean([np.mean(sample) for sample in self.cpu_samples])
        max_cpu = np.max([np.max(sample) for sample in self.cpu_samples])
        min_cpu = np.min([np.min(sample) for sample in self.cpu_samples])
        
        # CPU core utilization distribution
        if self.cpu_samples:
            cpu_core_avg = np.mean(self.cpu_samples, axis=0)
            cpu_core_max = np.max(self.cpu_samples, axis=0)
        else:
            cpu_core_avg = []
            cpu_core_max = []
        
        avg_ram_percent = np.mean([sample['percent'] for sample in self.ram_samples])
        max_ram_percent = np.max([sample['percent'] for sample in self.ram_samples])
        min_ram_percent = np.min([sample['percent'] for sample in self.ram_samples])
        
        # Get RAM usage in GB
        avg_ram_used = np.mean([sample['used'] for sample in self.ram_samples]) / (1024**3)
        max_ram_used = np.max([sample['used'] for sample in self.ram_samples]) / (1024**3)
        
        summary = {
            'cpu': {
                'available': True,
                'total_logical_cores': self.cpu_count,
                'total_physical_cores': self.physical_cpu_count,
                'avg_utilization_percent': float(avg_cpu),
                'max_utilization_percent': float(max_cpu),
                'min_utilization_percent': float(min_cpu),
                'core_utilization': {
                    'avg': [float(x) for x in cpu_core_avg],
                    'max': [float(x) for x in cpu_core_max]
                }
            },
            'ram': {
                'available': True,
                'total_gb': float(self.total_ram / (1024**3)),
                'avg_used_gb': float(avg_ram_used),
                'max_used_gb': float(max_ram_used),
                'avg_utilization_percent': float(avg_ram_percent),
                'max_utilization_percent': float(max_ram_percent),
                'min_utilization_percent': float(min_ram_percent)
            },
            'samples_taken': len(self.cpu_samples)
        }
        
        # Add GPU info if available
        if self.gpu_available and self.gpu_samples:
            gpu_summary = []
            
            for gpu_id in range(self.num_gpus):
                # Extract stats for this GPU ID from all samples
                gpu_utils = []
                gpu_mems = []
                gpu_temps = []
                
                for sample in self.gpu_samples:
                    if gpu_id < len(sample) and 'error' not in sample[gpu_id]:
                        if 'utilization' in sample[gpu_id]:
                            gpu_utils.append(sample[gpu_id]['utilization'])
                        if 'memory_percent' in sample[gpu_id]:
                            gpu_mems.append(sample[gpu_id]['memory_percent'])
                        if 'temperature' in sample[gpu_id]:
                            gpu_temps.append(sample[gpu_id]['temperature'])
                
                # Calculate stats if we have data
                if gpu_utils and gpu_mems:
                    # Get device name from first sample
                    device_name = self.gpu_samples[0][gpu_id].get('name', f"GPU #{gpu_id}")
                    memory_total_gb = self.gpu_samples[0][gpu_id].get('memory_total', 0) / (1024**3)
                    
                    gpu_summary.append({
                        'gpu_id': gpu_id,
                        'name': device_name,
                        'avg_utilization_percent': float(np.mean(gpu_utils)),
                        'max_utilization_percent': float(np.max(gpu_utils)),
                        'min_utilization_percent': float(np.min(gpu_utils)) if gpu_utils else 0,
                        'avg_memory_percent': float(np.mean(gpu_mems)),
                        'max_memory_percent': float(np.max(gpu_mems)),
                        'min_memory_percent': float(np.min(gpu_mems)) if gpu_mems else 0,
                        'avg_temperature': float(np.mean(gpu_temps)) if gpu_temps else 0,
                        'max_temperature': float(np.max(gpu_temps)) if gpu_temps else 0,
                        'memory_gb': float(memory_total_gb)
                    })
            
            summary['gpu'] = {
                'available': True,
                'count': self.num_gpus,
                'devices': gpu_summary
            }
        else:
            summary['gpu'] = {
                'available': False
            }
            
        return summary
        
    def reset(self):
        """Reset all samples"""
        self.cpu_samples = []
        self.ram_samples = []
        self.gpu_samples = []
        self.sample_timestamps = []

class PerformanceMonitor:
    """Class to track and log computation times for performance analysis"""
    
    def __init__(self, log_dir="logs"):
        """Initialize the performance monitor with given log directory"""
        self.log_dir = log_dir
        self.timings = {}
        self.current_pair = None
        self.lock = threading.Lock()
        
        # Create the log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a log file with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"computing_durations_{timestamp}.log")
        
        # Write header to log file
        with open(self.log_file, 'w') as f:
            f.write("Timestamp,Pair,Operation,Duration(s)\n")
            
        # Initialize resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.sampling_thread = None
        self.keep_sampling = False
    
    def start_resource_monitoring(self, interval=5.0):
        """Start periodic resource monitoring in a background thread"""
        self.keep_sampling = True
        
        def sampling_worker():
            while self.keep_sampling:
                try:
                    self.resource_monitor.take_sample()
                except Exception as e:
                    logging.warning(f"Error taking resource sample: {e}")
                time.sleep(interval)
        
        self.sampling_thread = threading.Thread(target=sampling_worker)
        self.sampling_thread.daemon = True
        self.sampling_thread.start()
        logging.info(f"Resource monitoring started with {interval}s interval")

    def stop_resource_monitoring(self):
        """Stop the resource monitoring thread"""
        self.keep_sampling = False
        if self.sampling_thread:
            self.sampling_thread.join(timeout=2.0)
            self.sampling_thread = None
            logging.info("Resource monitoring stopped")
    
    def start_pair(self, pair):
        """Set the current pair being processed"""
        with self.lock:
            self.current_pair = pair
            if pair not in self.timings:
                self.timings[pair] = {
                    "total": 0,
                    "operations": {}
                }
    
    def log_operation(self, operation, duration):
        """Log the duration of a specific operation"""
        with self.lock:
            if not self.current_pair:
                # Create a special entry for operations not associated with a pair
                if 'global' not in self.timings:
                    self.timings['global'] = {
                        "total": 0,
                        "operations": {}
                    }
                
                if operation not in self.timings['global']["operations"]:
                    self.timings['global']["operations"][operation] = []
                    
                self.timings['global']["operations"][operation].append(duration)
                
                # Write to log file immediately
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.log_file, 'a') as f:
                    f.write(f"{timestamp},global,{operation},{duration:.6f}\n")
                return
                
            if operation not in self.timings[self.current_pair]["operations"]:
                self.timings[self.current_pair]["operations"][operation] = []
                
            self.timings[self.current_pair]["operations"][operation].append(duration)
            
            # Write to log file immediately
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a') as f:
                f.write(f"{timestamp},{self.current_pair},{operation},{duration:.6f}\n")

    def end_pair(self, total_duration):
        """Log the total processing time for the current pair"""
        with self.lock:
            if not self.current_pair:
                # For global operations, or if no pair is set
                if 'global' in self.timings:
                    self.timings['global']["total"] += total_duration
                    
                    # Write total to log file
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(self.log_file, 'a') as f:
                        f.write(f"{timestamp},global,TOTAL,{total_duration:.6f}\n")
                return
                
            self.timings[self.current_pair]["total"] = total_duration
            
            # Write total to log file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a') as f:
                f.write(f"{timestamp},{self.current_pair},TOTAL,{total_duration:.6f}\n")
            
            # Reset current pair
            self.current_pair = None
    
    def save_summary(self):
        """Save a summary of all timings to JSON and text files"""
        # Stop resource monitoring before saving
        self.stop_resource_monitoring()
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        summary_file = os.path.join(self.log_dir, f"performance_summary_{timestamp}.json")
        
        # Check if we have any data
        if not self.timings:
            logging.warning("No performance data collected. Summary will be empty.")
            
        summary = {
            "pairs_processed": len(self.timings),
            "total_processing_time": sum(data["total"] for data in self.timings.values()),
            "average_pair_time": sum(data["total"] for data in self.timings.values()) / len(self.timings) if self.timings else 0,
            "operation_summaries": {}
        }
        
        # Calculate statistics for each operation
        all_operations = set()
        for pair_data in self.timings.values():
            all_operations.update(pair_data["operations"].keys())
        
        for operation in all_operations:
            operation_times = []
            for pair_data in self.timings.values():
                if operation in pair_data["operations"]:
                    operation_times.extend(pair_data["operations"][operation])
            
            if operation_times:
                summary["operation_summaries"][operation] = {
                    "total_calls": len(operation_times),
                    "average_time": sum(operation_times) / len(operation_times),
                    "min_time": min(operation_times),
                    "max_time": max(operation_times),
                    "total_time": sum(operation_times),
                    "percentage_of_total": (sum(operation_times) / summary["total_processing_time"]) * 100 if summary["total_processing_time"] > 0 else 0
                }
        
        # Sort operations by total time (descending)
        summary["operation_summaries"] = dict(
            sorted(
                summary["operation_summaries"].items(),
                key=lambda x: x[1]["total_time"],
                reverse=True
            )
        )
        
        # Add resource utilization data
        summary["resource_utilization"] = self.resource_monitor.get_summary()
        
        # Save to file
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save a readable text report
        report_file = os.path.join(self.log_dir, f"performance_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write("PERFORMANCE SUMMARY REPORT\n")
            f.write("=========================\n\n")
            f.write(f"Pairs Processed: {summary['pairs_processed']}\n")
            f.write(f"Total Processing Time: {summary['total_processing_time']:.2f} seconds\n")
            f.write(f"Average Time Per Pair: {summary['average_pair_time']:.2f} seconds\n\n")
            
            f.write("OPERATION BREAKDOWN (Sorted by Total Time)\n")
            f.write("----------------------------------------\n")
            f.write(f"{'Operation':<30} {'Total Time (s)':<15} {'Avg Time (s)':<15} {'Calls':<10} {'% of Total':<10}\n")
            f.write("-" * 80 + "\n")
            
            for op, stats in summary["operation_summaries"].items():
                f.write(
                    f"{op[:30]:<30} {stats['total_time']:<15.2f} {stats['average_time']:<15.6f} "
                    f"{stats['total_calls']:<10} {stats['percentage_of_total']:<10.2f}\n"
                )
            
            # Add resource utilization section
            resource_data = summary["resource_utilization"]
            f.write("\n\nRESOURCE UTILIZATION\n")
            f.write("===================\n\n")
            
            # CPU section
            cpu_info = resource_data.get("cpu", {})
            if cpu_info.get("available", False):
                f.write(f"CPU Cores: {cpu_info.get('total_physical_cores')} physical / {cpu_info.get('total_logical_cores')} logical\n")
                f.write(f"Average CPU Utilization: {cpu_info.get('avg_utilization_percent', 0):.2f}%\n")
                f.write(f"Maximum CPU Utilization: {cpu_info.get('max_utilization_percent', 0):.2f}%\n")
                f.write(f"Minimum CPU Utilization: {cpu_info.get('min_utilization_percent', 0):.2f}%\n\n")
                
                # Show per-core utilization if available
                core_util = cpu_info.get("core_utilization", {})
                if core_util and core_util.get("avg"):
                    f.write("Per-Core Utilization (Avg%):\n")
                    core_avgs = core_util.get("avg", [])
                    for i, avg in enumerate(core_avgs):
                        f.write(f"  Core {i}: {avg:.2f}%{' - Potential bottleneck' if avg > 90 else ''}\n")
                    f.write("\n")
            else:
                f.write("CPU monitoring data not available.\n\n")
            
            # RAM section
            ram_info = resource_data.get("ram", {})
            if ram_info.get("available", False):
                f.write(f"Total RAM: {ram_info.get('total_gb', 0):.2f} GB\n")
                f.write(f"Average RAM Used: {ram_info.get('avg_used_gb', 0):.2f} GB ({ram_info.get('avg_utilization_percent', 0):.2f}%)\n")
                f.write(f"Maximum RAM Used: {ram_info.get('max_used_gb', 0):.2f} GB ({ram_info.get('max_utilization_percent', 0):.2f}%)\n\n")
            else:
                f.write("RAM monitoring data not available.\n\n")
            
            # GPU section
            gpu_info = resource_data.get("gpu", {})
            if gpu_info.get("available", False):
                f.write(f"GPUs Available: {gpu_info.get('count', 0)}\n")
                for gpu in gpu_info.get("devices", []):
                    f.write(f"GPU #{gpu.get('gpu_id', '?')} - {gpu.get('name', 'Unknown')} ({gpu.get('memory_gb', 0):.2f} GB):\n")
                    f.write(f"  Average Utilization: {gpu.get('avg_utilization_percent', 0):.2f}%\n")
                    f.write(f"  Maximum Utilization: {gpu.get('max_utilization_percent', 0):.2f}%\n")
                    f.write(f"  Average Memory Usage: {gpu.get('avg_memory_percent', 0):.2f}%\n")
                    f.write(f"  Maximum Memory Usage: {gpu.get('max_memory_percent', 0):.2f}%\n")
                    if 'avg_temperature' in gpu:
                        f.write(f"  Average Temperature: {gpu.get('avg_temperature', 0):.2f}°C\n")
                    f.write("\n")
            else:
                f.write("No GPU utilization data available.\n\n")
            
            # Resource utilization analysis
            f.write("RESOURCE UTILIZATION ANALYSIS\n")
            f.write("----------------------------\n")
            
            # Calculate the number of samples for reference
            samples_taken = resource_data.get("samples_taken", 0)
            f.write(f"Resource samples taken: {samples_taken}\n\n")
            
            # CPU analysis
            if cpu_info.get("available", False):
                cpu_util = cpu_info.get('avg_utilization_percent', 0)
                if cpu_util < 50:
                    f.write("⚠️ CPU underutilized - consider increasing batch size or parallel processing\n")
                elif cpu_util > 90:
                    f.write("⚠️ CPU potentially overutilized - consider reducing batch size to avoid throttling\n")
                
                # Check for uneven core utilization
                if "core_utilization" in cpu_info and cpu_info["core_utilization"].get("avg"):
                    core_avgs = cpu_info["core_utilization"]["avg"]
                    max_util = max(core_avgs)
                    min_util = min(core_avgs)
                    if max_util - min_util > 30:  # Over 30% difference
                        f.write("⚠️ Significant core utilization imbalance detected - workload may not be properly distributed\n")
            
            # RAM analysis
            if ram_info.get("available", False):
                ram_util = ram_info.get('avg_utilization_percent', 0)
                ram_max = ram_info.get('max_utilization_percent', 0)
                
                if ram_util < 30:
                    f.write("⚠️ RAM underutilized - consider increasing batch size\n")
                elif ram_max > 90:
                    f.write("⚠️ RAM potentially overutilized - consider reducing batch size to avoid paging\n")
            
            # GPU analysis
            if gpu_info.get("available", False):
                gpu_devices = gpu_info.get("devices", [])
                for gpu in gpu_devices:
                    gpu_id = gpu.get('gpu_id', '?')
                    gpu_name = gpu.get('name', 'Unknown')
                    gpu_util = gpu.get('avg_utilization_percent', 0)
                    gpu_mem = gpu.get('avg_memory_percent', 0)
                    
                    if gpu_util < 30:
                        f.write(f"⚠️ GPU #{gpu_id} ({gpu_name}) underutilized ({gpu_util:.2f}%) - "
                               f"consider enabling more GPU-accelerated features\n")
                    if gpu_mem > 90:
                        f.write(f"⚠️ GPU #{gpu_id} ({gpu_name}) memory usage high ({gpu_mem:.2f}%) - "
                               f"consider batch processing to avoid memory issues\n")
                    
                    # Temperature warning if available
                    if 'max_temperature' in gpu and gpu['max_temperature'] > 80:
                        f.write(f"⚠️ GPU #{gpu_id} ({gpu_name}) temperature high ({gpu['max_temperature']:.2f}°C) - "
                               f"consider monitoring system cooling\n")
            
            # Overall efficiency analysis
            if cpu_info.get("available", False) and gpu_info.get("available", False):
                cpu_util = cpu_info.get('avg_utilization_percent', 0)
                gpu_devices = gpu_info.get("devices", [])
                if gpu_devices:
                    avg_gpu_util = sum(gpu.get('avg_utilization_percent', 0) for gpu in gpu_devices) / len(gpu_devices)
                    
                    if cpu_util > 80 and avg_gpu_util < 30:
                        f.write("\n⚠️ CPU-bound processing detected - GPU acceleration not effectively utilized\n")
                    if cpu_util < 50 and avg_gpu_util > 80:
                        f.write("\n⚠️ GPU-bound processing detected - CPU may be waiting for GPU operations\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("---------------\n")
            
            # Generate overall recommendations based on resource usage patterns
            recommendations = []
            
            # Check if numba/GPU could help
            if cpu_info.get("available", False) and cpu_info.get('avg_utilization_percent', 0) > 80:
                if not gpu_info.get("available", False) or gpu_info.get("count", 0) == 0:
                    recommendations.append("Consider enabling Numba JIT optimization for CPU-intensive calculations")
                    recommendations.append("Consider adding GPU acceleration if hardware is available")
                elif gpu_info.get("available", False):
                    gpu_devices = gpu_info.get("devices", [])
                    if gpu_devices and any(gpu.get('avg_utilization_percent', 0) < 50 for gpu in gpu_devices):
                        recommendations.append("GPU is underutilized while CPU is heavily loaded - "
                                              "optimize more operations for GPU acceleration")
            
            # Check batch size recommendations
            if (cpu_info.get("available", False) and cpu_info.get('avg_utilization_percent', 0) < 50 and
                ram_info.get("available", False) and ram_info.get('avg_utilization_percent', 0) < 60):
                recommendations.append("Consider increasing batch size to better utilize available resources")
            
            # Provide relevant performance tips
            recommendations.append("Review the slowest operations (top of the table) for potential optimization")
            
            # Print recommendations
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            
        logging.info(f"Performance summary saved to {summary_file}")
        logging.info(f"Performance report saved to {report_file}")
        
        # Reset resource monitor
        self.resource_monitor.reset()
        
        return summary_file, report_file