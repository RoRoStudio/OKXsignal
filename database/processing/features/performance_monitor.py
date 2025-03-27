# database/processing/features/performance_monitor.py
#!/usr/bin/env python3
"""
Performance Monitoring for feature computation
"""

import os
import logging
import threading
import time
from datetime import datetime

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
            # Check if current_pair is None before accessing
            if self.current_pair is None:
                logging.warning(f"Attempted to log operation '{operation}' but current_pair is None")
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
            # Check if current_pair is None before accessing
            if self.current_pair is None:
                logging.warning(f"Attempted to end timing but current_pair is None")
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
        import json
        
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
        
        return summary_file, report_file