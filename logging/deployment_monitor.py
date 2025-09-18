#!/usr/bin/env python3
"""
Deployment Monitoring Script
Tracks deployment progress, health checks, and system status.
"""

import os
import sys
import time
import json
import subprocess
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import signal

# Add the parent directory to the path to import our logger
sys.path.append(str(Path(__file__).parent.parent))

try:
    from logging.comprehensive_logger import comprehensive_logger
except ImportError:
    # Fallback to basic logging if comprehensive logger is not available
    import logging
    logging.basicConfig(level=logging.INFO)
    comprehensive_logger = logging.getLogger(__name__)


class DeploymentMonitor:
    """Monitor deployment process and system health"""
    
    def __init__(self):
        self.logger = comprehensive_logger
        self.monitoring = False
        self.health_check_interval = 30  # seconds
        self.deployment_stages = [
            "pre_deployment_checks",
            "docker_verification",
            "file_transfer",
            "deployment_execution",
            "post_deployment_validation",
            "health_monitoring"
        ]
        self.current_stage = None
        self.deployment_start_time = None
        
    def start_monitoring(self):
        """Start the deployment monitoring process"""
        self.monitoring = True
        self.deployment_start_time = datetime.now(timezone.utc)
        
        self.logger.log_deployment(
            "monitoring", "STARTED", 
            "Deployment monitoring initiated",
            {"start_time": self.deployment_start_time.isoformat()}
        )
        
        # Start health check thread
        health_thread = threading.Thread(target=self._continuous_health_check)
        health_thread.daemon = True
        health_thread.start()
        
    def stop_monitoring(self):
        """Stop the deployment monitoring process"""
        self.monitoring = False
        duration = (datetime.now(timezone.utc) - self.deployment_start_time).total_seconds()
        
        self.logger.log_deployment(
            "monitoring", "STOPPED", 
            "Deployment monitoring completed",
            {"duration_seconds": duration}
        )
        
    def log_stage_start(self, stage: str, details: Optional[Dict[str, Any]] = None):
        """Log the start of a deployment stage"""
        self.current_stage = stage
        self.logger.log_deployment(
            stage, "STARTED", 
            f"Starting deployment stage: {stage}",
            details or {}
        )
        
    def log_stage_complete(self, stage: str, success: bool = True, 
                          details: Optional[Dict[str, Any]] = None):
        """Log the completion of a deployment stage"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.log_deployment(
            stage, status, 
            f"Deployment stage {stage} {status.lower()}",
            details or {}
        )
        
    def check_docker_status(self) -> Dict[str, Any]:
        """Check Docker installation and service status"""
        self.log_stage_start("docker_verification")
        
        try:
            # Check if Docker is installed
            docker_version = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, text=True, timeout=10
            )
            
            if docker_version.returncode != 0:
                self.log_stage_complete("docker_verification", False, {
                    "error": "Docker not installed or not accessible"
                })
                return {"status": "failed", "error": "Docker not installed"}
            
            # Check if Docker daemon is running
            docker_info = subprocess.run(
                ["docker", "info"], 
                capture_output=True, text=True, timeout=10
            )
            
            if docker_info.returncode != 0:
                self.log_stage_complete("docker_verification", False, {
                    "error": "Docker daemon not running"
                })
                return {"status": "failed", "error": "Docker daemon not running"}
            
            # Get Docker system information
            docker_ps = subprocess.run(
                ["docker", "ps", "--format", "json"], 
                capture_output=True, text=True, timeout=10
            )
            
            running_containers = []
            if docker_ps.returncode == 0 and docker_ps.stdout.strip():
                for line in docker_ps.stdout.strip().split('\n'):
                    try:
                        container = json.loads(line)
                        running_containers.append(container)
                    except json.JSONDecodeError:
                        pass
            
            result = {
                "status": "success",
                "version": docker_version.stdout.strip(),
                "running_containers": len(running_containers),
                "containers": running_containers
            }
            
            self.log_stage_complete("docker_verification", True, result)
            return result
            
        except subprocess.TimeoutExpired:
            error_msg = "Docker command timed out"
            self.log_stage_complete("docker_verification", False, {"error": error_msg})
            return {"status": "failed", "error": error_msg}
        except Exception as e:
            error_msg = f"Error checking Docker status: {str(e)}"
            self.log_stage_complete("docker_verification", False, {"error": error_msg})
            return {"status": "failed", "error": error_msg}
    
    def monitor_file_transfer(self, source_path: str, destination: str) -> Dict[str, Any]:
        """Monitor file transfer process"""
        self.log_stage_start("file_transfer", {
            "source": source_path,
            "destination": destination
        })
        
        try:
            # Get source directory size
            source_size = self._get_directory_size(source_path)
            
            # Log transfer initiation
            self.logger.log_deployment(
                "file_transfer", "IN_PROGRESS",
                f"Transferring {source_size} bytes from {source_path} to {destination}"
            )
            
            # This would be where actual file transfer monitoring happens
            # For now, we'll simulate the monitoring
            result = {
                "status": "success",
                "source_size": source_size,
                "transfer_method": "rsync/scp",
                "destination": destination
            }
            
            self.log_stage_complete("file_transfer", True, result)
            return result
            
        except Exception as e:
            error_msg = f"Error monitoring file transfer: {str(e)}"
            self.log_stage_complete("file_transfer", False, {"error": error_msg})
            return {"status": "failed", "error": error_msg}
    
    def monitor_deployment_script(self, script_path: str) -> Dict[str, Any]:
        """Monitor the execution of the deployment script"""
        self.log_stage_start("deployment_execution", {"script": script_path})
        
        try:
            start_time = datetime.now()
            
            # Execute the deployment script
            process = subprocess.Popen(
                [script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(script_path)
            )
            
            # Monitor the process
            stdout, stderr = process.communicate()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                "status": "success" if process.returncode == 0 else "failed",
                "return_code": process.returncode,
                "duration_seconds": duration,
                "stdout_lines": len(stdout.split('\n')) if stdout else 0,
                "stderr_lines": len(stderr.split('\n')) if stderr else 0
            }
            
            if process.returncode == 0:
                self.log_stage_complete("deployment_execution", True, result)
            else:
                result["error"] = stderr[:500] if stderr else "Unknown error"
                self.log_stage_complete("deployment_execution", False, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error monitoring deployment script: {str(e)}"
            self.log_stage_complete("deployment_execution", False, {"error": error_msg})
            return {"status": "failed", "error": error_msg}
    
    def perform_health_check(self, url: str = "http://localhost:8000") -> Dict[str, Any]:
        """Perform application health check"""
        try:
            start_time = time.time()
            response = requests.get(f"{url}/health", timeout=10)
            response_time = time.time() - start_time
            
            result = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time": response_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    result["health_data"] = health_data
                except json.JSONDecodeError:
                    result["health_data"] = {"response": response.text[:200]}
            
            return result
            
        except requests.RequestException as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _continuous_health_check(self):
        """Continuously perform health checks while monitoring is active"""
        while self.monitoring:
            health_result = self.perform_health_check()
            
            if health_result["status"] == "healthy":
                self.logger.log_deployment(
                    "health_check", "SUCCESS",
                    "Application health check passed",
                    health_result
                )
            else:
                self.logger.log_deployment(
                    "health_check", "WARNING",
                    "Application health check failed",
                    health_result
                )
            
            time.sleep(self.health_check_interval)
    
    def _get_directory_size(self, path: str) -> int:
        """Get the total size of a directory"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate a comprehensive deployment report"""
        if not self.deployment_start_time:
            return {"error": "No deployment monitoring session active"}
        
        duration = (datetime.now(timezone.utc) - self.deployment_start_time).total_seconds()
        
        report = {
            "deployment_summary": {
                "start_time": self.deployment_start_time.isoformat(),
                "duration_seconds": duration,
                "current_stage": self.current_stage,
                "monitoring_active": self.monitoring
            },
            "system_status": {
                "docker": self.check_docker_status(),
                "health": self.perform_health_check()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return report


def main():
    """Main function for running deployment monitoring"""
    monitor = DeploymentMonitor()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\nShutting down deployment monitor...")
        monitor.stop_monitoring()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Keep the monitor running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()