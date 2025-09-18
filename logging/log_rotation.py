#!/usr/bin/env python3
"""
Log Rotation and Cleanup System
Manages log file rotation, compression, and cleanup to prevent disk space issues.
"""

import os
import gzip
import shutil
import glob
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
from typing import List, Dict, Any


class LogRotationManager:
    """Manages log file rotation and cleanup"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_dir = Path(log_directory)
        self.config = {
            "max_file_size_mb": 10,
            "max_files_per_type": 5,
            "compress_after_days": 1,
            "delete_after_days": 30,
            "log_types": [
                "application",
                "errors",
                "audit", 
                "deployment",
                "performance",
                "security"
            ]
        }
        
        # Set up rotation logger
        self.logger = logging.getLogger("log_rotation")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def rotate_logs(self) -> Dict[str, Any]:
        """Perform log rotation for all log types"""
        rotation_summary = {
            "timestamp": datetime.now().isoformat(),
            "rotated_files": [],
            "compressed_files": [],
            "deleted_files": [],
            "errors": []
        }
        
        try:
            # Ensure log directory exists
            self.log_dir.mkdir(exist_ok=True)
            
            # Rotate logs for each type
            for log_type in self.config["log_types"]:
                try:
                    result = self._rotate_log_type(log_type)
                    rotation_summary["rotated_files"].extend(result.get("rotated", []))
                    rotation_summary["compressed_files"].extend(result.get("compressed", []))
                    rotation_summary["deleted_files"].extend(result.get("deleted", []))
                except Exception as e:
                    error_msg = f"Error rotating {log_type} logs: {str(e)}"
                    self.logger.error(error_msg)
                    rotation_summary["errors"].append(error_msg)
            
            # Compress old log files
            self._compress_old_logs(rotation_summary)
            
            # Clean up very old files
            self._cleanup_old_files(rotation_summary)
            
            # Log rotation summary
            self.logger.info(f"Log rotation completed: {json.dumps(rotation_summary, indent=2)}")
            
        except Exception as e:
            error_msg = f"Critical error during log rotation: {str(e)}"
            self.logger.error(error_msg)
            rotation_summary["errors"].append(error_msg)
        
        return rotation_summary
    
    def _rotate_log_type(self, log_type: str) -> Dict[str, List[str]]:
        """Rotate logs for a specific log type"""
        result = {"rotated": [], "compressed": [], "deleted": []}
        
        # Find log files for this type
        log_patterns = [
            self.log_dir / f"{log_type}.log",
            self.log_dir / log_type / f"{log_type}.log",
            self.log_dir / f"{log_type}*.log"
        ]
        
        for pattern in log_patterns:
            log_files = glob.glob(str(pattern))
            
            for log_file_path in log_files:
                log_file = Path(log_file_path)
                
                if not log_file.exists():
                    continue
                
                # Check if file needs rotation
                if self._needs_rotation(log_file):
                    rotated_file = self._perform_rotation(log_file)
                    if rotated_file:
                        result["rotated"].append(str(rotated_file))
                        self.logger.info(f"Rotated {log_file} to {rotated_file}")
        
        return result
    
    def _needs_rotation(self, log_file: Path) -> bool:
        """Check if a log file needs rotation"""
        try:
            # Check file size
            file_size_mb = log_file.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config["max_file_size_mb"]:
                return True
            
            # Check file age (rotate daily)
            file_age = datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)
            if file_age > timedelta(days=1):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking if {log_file} needs rotation: {str(e)}")
            return False
    
    def _perform_rotation(self, log_file: Path) -> Path:
        """Perform the actual rotation of a log file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_name = f"{log_file.stem}_{timestamp}.log"
            rotated_path = log_file.parent / rotated_name
            
            # Move the current log file
            shutil.move(str(log_file), str(rotated_path))
            
            # Create a new empty log file
            log_file.touch()
            
            return rotated_path
            
        except Exception as e:
            self.logger.error(f"Error rotating {log_file}: {str(e)}")
            return None
    
    def _compress_old_logs(self, summary: Dict[str, Any]):
        """Compress log files older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config["compress_after_days"])
            
            # Find all .log files (excluding current ones)
            for log_file in self.log_dir.rglob("*.log"):
                if log_file.name.endswith("_*.log") or "_" in log_file.stem:
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    
                    if file_mtime < cutoff_date and not str(log_file).endswith(".gz"):
                        compressed_file = self._compress_file(log_file)
                        if compressed_file:
                            summary["compressed_files"].append(str(compressed_file))
                            self.logger.info(f"Compressed {log_file} to {compressed_file}")
                            
        except Exception as e:
            error_msg = f"Error compressing old logs: {str(e)}"
            self.logger.error(error_msg)
            summary["errors"].append(error_msg)
    
    def _compress_file(self, file_path: Path) -> Path:
        """Compress a single file using gzip"""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file after successful compression
            file_path.unlink()
            
            return compressed_path
            
        except Exception as e:
            self.logger.error(f"Error compressing {file_path}: {str(e)}")
            return None
    
    def _cleanup_old_files(self, summary: Dict[str, Any]):
        """Delete very old log files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config["delete_after_days"])
            
            # Find all compressed log files
            for log_file in self.log_dir.rglob("*.log.gz"):
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    try:
                        log_file.unlink()
                        summary["deleted_files"].append(str(log_file))
                        self.logger.info(f"Deleted old log file: {log_file}")
                    except Exception as e:
                        error_msg = f"Error deleting {log_file}: {str(e)}"
                        self.logger.error(error_msg)
                        summary["errors"].append(error_msg)
                        
        except Exception as e:
            error_msg = f"Error cleaning up old files: {str(e)}"
            self.logger.error(error_msg)
            summary["errors"].append(error_msg)
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get statistics about log files"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_files": 0,
            "total_size_mb": 0,
            "by_type": {},
            "oldest_file": None,
            "newest_file": None
        }
        
        try:
            oldest_time = None
            newest_time = None
            
            for log_file in self.log_dir.rglob("*.log*"):
                if log_file.is_file():
                    stats["total_files"] += 1
                    file_size = log_file.stat().st_size / (1024 * 1024)
                    stats["total_size_mb"] += file_size
                    
                    # Track by type
                    log_type = log_file.parent.name if log_file.parent != self.log_dir else "general"
                    if log_type not in stats["by_type"]:
                        stats["by_type"][log_type] = {"files": 0, "size_mb": 0}
                    
                    stats["by_type"][log_type]["files"] += 1
                    stats["by_type"][log_type]["size_mb"] += file_size
                    
                    # Track oldest and newest
                    file_mtime = log_file.stat().st_mtime
                    if oldest_time is None or file_mtime < oldest_time:
                        oldest_time = file_mtime
                        stats["oldest_file"] = str(log_file)
                    
                    if newest_time is None or file_mtime > newest_time:
                        newest_time = file_mtime
                        stats["newest_file"] = str(log_file)
            
            # Round total size
            stats["total_size_mb"] = round(stats["total_size_mb"], 2)
            
            # Round sizes by type
            for log_type in stats["by_type"]:
                stats["by_type"][log_type]["size_mb"] = round(
                    stats["by_type"][log_type]["size_mb"], 2
                )
                
        except Exception as e:
            self.logger.error(f"Error getting log statistics: {str(e)}")
            stats["error"] = str(e)
        
        return stats


def main():
    """Main function for running log rotation"""
    rotation_manager = LogRotationManager()
    
    print("Starting log rotation...")
    result = rotation_manager.rotate_logs()
    
    print("\nLog Rotation Summary:")
    print(json.dumps(result, indent=2))
    
    print("\nLog Statistics:")
    stats = rotation_manager.get_log_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()