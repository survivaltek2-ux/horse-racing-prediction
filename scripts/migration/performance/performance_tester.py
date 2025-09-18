#!/usr/bin/env python3
"""
PostgreSQL Migration Performance Tester
Comprehensive performance testing and validation for migrated database
"""

import os
import sys
import json
import logging
import time
import statistics
import threading
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import pandas as pd

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.database_config import DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/performance_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    test_name: str
    operation: str
    duration_ms: float
    rows_affected: int
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

@dataclass
class PerformanceTestResult:
    """Results of a performance test suite"""
    test_suite: str
    database_type: str
    start_time: datetime
    end_time: datetime
    total_duration_ms: float
    metrics: List[PerformanceMetric]
    summary_stats: Dict[str, Any]
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_suite': self.test_suite,
            'database_type': self.database_type,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_duration_ms': self.total_duration_ms,
            'metrics': [asdict(m) for m in self.metrics],
            'summary_stats': self.summary_stats,
            'success': self.success
        }

class PerformanceTester:
    """Comprehensive performance testing for PostgreSQL migration"""
    
    def __init__(self, source_db_uri: str, target_db_uri: str):
        self.source_db_uri = source_db_uri
        self.target_db_uri = target_db_uri
        
        # Create performance testing directory
        self.perf_dir = Path("scripts/migration/performance")
        self.perf_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.perf_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.charts_dir = self.perf_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)
        
        # Database connections
        self.source_engine = create_engine(source_db_uri)
        self.target_engine = create_engine(target_db_uri)
        
        # Test configurations
        self.test_configs = self._load_test_configs()
        
        # Results storage
        self.test_results = []
    
    def _load_test_configs(self) -> Dict[str, Any]:
        """Load performance test configurations"""
        default_configs = {
            "connection_tests": {
                "max_connections": 50,
                "connection_timeout": 30,
                "test_duration": 60
            },
            "query_tests": {
                "simple_select_iterations": 1000,
                "complex_join_iterations": 100,
                "aggregation_iterations": 500,
                "concurrent_queries": 10
            },
            "crud_tests": {
                "insert_batch_size": 1000,
                "update_batch_size": 500,
                "delete_batch_size": 100,
                "concurrent_operations": 5
            },
            "load_tests": {
                "duration_minutes": 10,
                "concurrent_users": 20,
                "ramp_up_time": 60
            },
            "stress_tests": {
                "max_concurrent_connections": 100,
                "large_dataset_size": 100000,
                "memory_pressure_mb": 1000
            }
        }
        
        config_file = self.perf_dir / "test_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    configs = json.load(f)
                    # Merge with defaults
                    for key, value in default_configs.items():
                        if key not in configs:
                            configs[key] = value
                    return configs
            except Exception as e:
                logger.warning(f"Failed to load test config, using defaults: {e}")
        
        # Save default config
        with open(config_file, 'w') as f:
            json.dump(default_configs, f, indent=2)
        
        return default_configs
    
    def _get_system_metrics(self) -> Tuple[float, float]:
        """Get current system metrics (memory, CPU)"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            return memory_mb, cpu_percent
        except ImportError:
            logger.warning("psutil not available, using default metrics")
            return 0.0, 0.0
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return 0.0, 0.0
    
    def _execute_timed_query(self, engine, query: str, params: Dict = None) -> PerformanceMetric:
        """Execute a query and measure performance"""
        start_time = time.time()
        memory_before, cpu_before = self._get_system_metrics()
        
        try:
            with engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                # Count rows if it's a SELECT
                rows_affected = 0
                if query.strip().upper().startswith('SELECT'):
                    rows_affected = len(result.fetchall())
                else:
                    rows_affected = result.rowcount
                
                conn.commit()
            
            end_time = time.time()
            memory_after, cpu_after = self._get_system_metrics()
            
            duration_ms = (end_time - start_time) * 1000
            memory_usage = max(0, memory_after - memory_before)
            cpu_usage = max(0, cpu_after - cpu_before)
            
            return PerformanceMetric(
                test_name="query_execution",
                operation=query[:50] + "..." if len(query) > 50 else query,
                duration_ms=duration_ms,
                rows_affected=rows_affected,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                timestamp=datetime.now(),
                success=True
            )
            
        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            return PerformanceMetric(
                test_name="query_execution",
                operation=query[:50] + "..." if len(query) > 50 else query,
                duration_ms=duration_ms,
                rows_affected=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    def test_connection_performance(self, database_type: str) -> PerformanceTestResult:
        """Test database connection performance"""
        logger.info(f"Testing connection performance for {database_type}")
        
        engine = self.target_engine if database_type == "postgresql" else self.source_engine
        config = self.test_configs["connection_tests"]
        
        start_time = datetime.now()
        metrics = []
        
        # Test single connection time
        for i in range(10):
            conn_start = time.time()
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                conn_end = time.time()
                
                metrics.append(PerformanceMetric(
                    test_name="single_connection",
                    operation="connect_and_query",
                    duration_ms=(conn_end - conn_start) * 1000,
                    rows_affected=1,
                    memory_usage_mb=0,
                    cpu_usage_percent=0,
                    timestamp=datetime.now(),
                    success=True
                ))
            except Exception as e:
                metrics.append(PerformanceMetric(
                    test_name="single_connection",
                    operation="connect_and_query",
                    duration_ms=0,
                    rows_affected=0,
                    memory_usage_mb=0,
                    cpu_usage_percent=0,
                    timestamp=datetime.now(),
                    success=False,
                    error_message=str(e)
                ))
        
        # Test concurrent connections
        def test_concurrent_connection():
            conn_start = time.time()
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    time.sleep(0.1)  # Hold connection briefly
                conn_end = time.time()
                return (conn_end - conn_start) * 1000, True, None
            except Exception as e:
                return 0, False, str(e)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["max_connections"]) as executor:
            futures = [executor.submit(test_concurrent_connection) for _ in range(config["max_connections"])]
            
            for future in concurrent.futures.as_completed(futures):
                duration, success, error = future.result()
                metrics.append(PerformanceMetric(
                    test_name="concurrent_connections",
                    operation="concurrent_connect",
                    duration_ms=duration,
                    rows_affected=1 if success else 0,
                    memory_usage_mb=0,
                    cpu_usage_percent=0,
                    timestamp=datetime.now(),
                    success=success,
                    error_message=error
                ))
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() * 1000
        
        # Calculate summary statistics
        successful_metrics = [m for m in metrics if m.success]
        summary_stats = {
            "total_tests": len(metrics),
            "successful_tests": len(successful_metrics),
            "failed_tests": len(metrics) - len(successful_metrics),
            "avg_connection_time_ms": statistics.mean([m.duration_ms for m in successful_metrics]) if successful_metrics else 0,
            "min_connection_time_ms": min([m.duration_ms for m in successful_metrics]) if successful_metrics else 0,
            "max_connection_time_ms": max([m.duration_ms for m in successful_metrics]) if successful_metrics else 0,
            "success_rate": len(successful_metrics) / len(metrics) * 100 if metrics else 0
        }
        
        return PerformanceTestResult(
            test_suite="connection_performance",
            database_type=database_type,
            start_time=start_time,
            end_time=end_time,
            total_duration_ms=total_duration,
            metrics=metrics,
            summary_stats=summary_stats,
            success=summary_stats["success_rate"] > 95
        )
    
    def test_query_performance(self, database_type: str) -> PerformanceTestResult:
        """Test query performance"""
        logger.info(f"Testing query performance for {database_type}")
        
        engine = self.target_engine if database_type == "postgresql" else self.source_engine
        config = self.test_configs["query_tests"]
        
        start_time = datetime.now()
        metrics = []
        
        # Define test queries
        test_queries = [
            # Simple SELECT queries
            ("simple_select", "SELECT * FROM races LIMIT 100"),
            ("simple_count", "SELECT COUNT(*) FROM horses"),
            ("simple_filter", "SELECT * FROM predictions WHERE confidence > 0.7"),
            
            # Complex JOIN queries
            ("complex_join", """
                SELECT r.race_name, h.horse_name, p.prediction_value, p.confidence
                FROM races r
                JOIN predictions p ON r.id = p.race_id
                JOIN horses h ON p.horse_id = h.id
                WHERE r.race_date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY p.confidence DESC
                LIMIT 50
            """),
            
            # Aggregation queries
            ("aggregation", """
                SELECT 
                    h.horse_name,
                    COUNT(p.id) as prediction_count,
                    AVG(p.confidence) as avg_confidence,
                    MAX(p.prediction_value) as max_prediction
                FROM horses h
                LEFT JOIN predictions p ON h.id = p.horse_id
                GROUP BY h.id, h.horse_name
                HAVING COUNT(p.id) > 0
                ORDER BY avg_confidence DESC
            """),
            
            # Subquery tests
            ("subquery", """
                SELECT * FROM races 
                WHERE id IN (
                    SELECT DISTINCT race_id 
                    FROM predictions 
                    WHERE confidence > (
                        SELECT AVG(confidence) FROM predictions
                    )
                )
            """)
        ]
        
        # Execute each query multiple times
        for query_name, query in test_queries:
            iterations = config.get(f"{query_name}_iterations", config["simple_select_iterations"])
            
            for i in range(iterations):
                metric = self._execute_timed_query(engine, query)
                metric.test_name = f"{query_name}_performance"
                metrics.append(metric)
        
        # Test concurrent queries
        def execute_concurrent_query(query_name, query):
            return self._execute_timed_query(engine, query)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["concurrent_queries"]) as executor:
            # Submit concurrent queries
            futures = []
            for _ in range(config["concurrent_queries"]):
                for query_name, query in test_queries[:3]:  # Use first 3 queries for concurrency test
                    future = executor.submit(execute_concurrent_query, query_name, query)
                    futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                metric = future.result()
                metric.test_name = f"concurrent_{metric.test_name}"
                metrics.append(metric)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() * 1000
        
        # Calculate summary statistics
        successful_metrics = [m for m in metrics if m.success]
        summary_stats = {
            "total_queries": len(metrics),
            "successful_queries": len(successful_metrics),
            "failed_queries": len(metrics) - len(successful_metrics),
            "avg_query_time_ms": statistics.mean([m.duration_ms for m in successful_metrics]) if successful_metrics else 0,
            "min_query_time_ms": min([m.duration_ms for m in successful_metrics]) if successful_metrics else 0,
            "max_query_time_ms": max([m.duration_ms for m in successful_metrics]) if successful_metrics else 0,
            "queries_per_second": len(successful_metrics) / (total_duration / 1000) if total_duration > 0 else 0,
            "success_rate": len(successful_metrics) / len(metrics) * 100 if metrics else 0
        }
        
        return PerformanceTestResult(
            test_suite="query_performance",
            database_type=database_type,
            start_time=start_time,
            end_time=end_time,
            total_duration_ms=total_duration,
            metrics=metrics,
            summary_stats=summary_stats,
            success=summary_stats["success_rate"] > 95
        )
    
    def test_crud_performance(self, database_type: str) -> PerformanceTestResult:
        """Test CRUD operation performance"""
        logger.info(f"Testing CRUD performance for {database_type}")
        
        engine = self.target_engine if database_type == "postgresql" else self.source_engine
        config = self.test_configs["crud_tests"]
        
        start_time = datetime.now()
        metrics = []
        
        # Test INSERT performance
        insert_query = """
            INSERT INTO horses (horse_name, age, weight, jockey, trainer, owner, created_at)
            VALUES (:name, :age, :weight, :jockey, :trainer, :owner, :created_at)
        """
        
        # Batch INSERT test
        insert_start = time.time()
        try:
            with engine.connect() as conn:
                for i in range(config["insert_batch_size"]):
                    conn.execute(text(insert_query), {
                        'name': f'Test Horse {i}',
                        'age': 3 + (i % 5),
                        'weight': 450 + (i % 100),
                        'jockey': f'Jockey {i % 10}',
                        'trainer': f'Trainer {i % 5}',
                        'owner': f'Owner {i % 20}',
                        'created_at': datetime.now()
                    })
                conn.commit()
            
            insert_end = time.time()
            metrics.append(PerformanceMetric(
                test_name="batch_insert",
                operation="INSERT horses",
                duration_ms=(insert_end - insert_start) * 1000,
                rows_affected=config["insert_batch_size"],
                memory_usage_mb=0,
                cpu_usage_percent=0,
                timestamp=datetime.now(),
                success=True
            ))
        except Exception as e:
            metrics.append(PerformanceMetric(
                test_name="batch_insert",
                operation="INSERT horses",
                duration_ms=0,
                rows_affected=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            ))
        
        # Test UPDATE performance
        update_query = """
            UPDATE horses 
            SET weight = weight + :weight_change 
            WHERE horse_name LIKE :name_pattern
        """
        
        for i in range(config["update_batch_size"]):
            metric = self._execute_timed_query(engine, update_query, {
                'weight_change': i % 10,
                'name_pattern': f'Test Horse {i}%'
            })
            metric.test_name = "update_performance"
            metrics.append(metric)
        
        # Test DELETE performance
        delete_query = "DELETE FROM horses WHERE horse_name LIKE :name_pattern"
        
        for i in range(config["delete_batch_size"]):
            metric = self._execute_timed_query(engine, delete_query, {
                'name_pattern': f'Test Horse {i}%'
            })
            metric.test_name = "delete_performance"
            metrics.append(metric)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() * 1000
        
        # Calculate summary statistics
        successful_metrics = [m for m in metrics if m.success]
        summary_stats = {
            "total_operations": len(metrics),
            "successful_operations": len(successful_metrics),
            "failed_operations": len(metrics) - len(successful_metrics),
            "avg_operation_time_ms": statistics.mean([m.duration_ms for m in successful_metrics]) if successful_metrics else 0,
            "operations_per_second": len(successful_metrics) / (total_duration / 1000) if total_duration > 0 else 0,
            "success_rate": len(successful_metrics) / len(metrics) * 100 if metrics else 0
        }
        
        return PerformanceTestResult(
            test_suite="crud_performance",
            database_type=database_type,
            start_time=start_time,
            end_time=end_time,
            total_duration_ms=total_duration,
            metrics=metrics,
            summary_stats=summary_stats,
            success=summary_stats["success_rate"] > 95
        )
    
    def test_load_performance(self, database_type: str) -> PerformanceTestResult:
        """Test database performance under load"""
        logger.info(f"Testing load performance for {database_type}")
        
        engine = self.target_engine if database_type == "postgresql" else self.source_engine
        config = self.test_configs["load_tests"]
        
        start_time = datetime.now()
        metrics = []
        
        # Define load test queries
        load_queries = [
            "SELECT COUNT(*) FROM races",
            "SELECT * FROM horses ORDER BY created_at DESC LIMIT 10",
            "SELECT AVG(confidence) FROM predictions WHERE confidence > 0.5",
            "SELECT r.race_name, COUNT(p.id) FROM races r LEFT JOIN predictions p ON r.id = p.race_id GROUP BY r.id, r.race_name"
        ]
        
        def load_test_worker(worker_id: int, duration_seconds: int):
            """Worker function for load testing"""
            worker_metrics = []
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                for query in load_queries:
                    metric = self._execute_timed_query(engine, query)
                    metric.test_name = f"load_test_worker_{worker_id}"
                    worker_metrics.append(metric)
                    
                    # Small delay to prevent overwhelming the database
                    time.sleep(0.01)
            
            return worker_metrics
        
        # Run load test with multiple workers
        duration_seconds = config["duration_minutes"] * 60
        concurrent_users = config["concurrent_users"]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(load_test_worker, i, duration_seconds)
                for i in range(concurrent_users)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                worker_metrics = future.result()
                metrics.extend(worker_metrics)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() * 1000
        
        # Calculate summary statistics
        successful_metrics = [m for m in metrics if m.success]
        summary_stats = {
            "total_queries": len(metrics),
            "successful_queries": len(successful_metrics),
            "failed_queries": len(metrics) - len(successful_metrics),
            "avg_response_time_ms": statistics.mean([m.duration_ms for m in successful_metrics]) if successful_metrics else 0,
            "median_response_time_ms": statistics.median([m.duration_ms for m in successful_metrics]) if successful_metrics else 0,
            "p95_response_time_ms": statistics.quantiles([m.duration_ms for m in successful_metrics], n=20)[18] if len(successful_metrics) > 20 else 0,
            "queries_per_second": len(successful_metrics) / (total_duration / 1000) if total_duration > 0 else 0,
            "concurrent_users": concurrent_users,
            "test_duration_minutes": config["duration_minutes"],
            "success_rate": len(successful_metrics) / len(metrics) * 100 if metrics else 0
        }
        
        return PerformanceTestResult(
            test_suite="load_performance",
            database_type=database_type,
            start_time=start_time,
            end_time=end_time,
            total_duration_ms=total_duration,
            metrics=metrics,
            summary_stats=summary_stats,
            success=summary_stats["success_rate"] > 90 and summary_stats["avg_response_time_ms"] < 1000
        )
    
    def run_comprehensive_performance_tests(self) -> Dict[str, PerformanceTestResult]:
        """Run all performance tests"""
        logger.info("Starting comprehensive performance tests")
        
        test_results = {}
        
        # Test both databases
        for db_type in ["sqlite", "postgresql"]:
            logger.info(f"Testing {db_type} database")
            
            try:
                # Connection performance
                conn_result = self.test_connection_performance(db_type)
                test_results[f"{db_type}_connection"] = conn_result
                
                # Query performance
                query_result = self.test_query_performance(db_type)
                test_results[f"{db_type}_query"] = query_result
                
                # CRUD performance
                crud_result = self.test_crud_performance(db_type)
                test_results[f"{db_type}_crud"] = crud_result
                
                # Load performance
                load_result = self.test_load_performance(db_type)
                test_results[f"{db_type}_load"] = load_result
                
            except Exception as e:
                logger.error(f"Failed to test {db_type} database: {e}")
        
        # Save results
        self._save_test_results(test_results)
        
        # Generate reports
        self._generate_performance_report(test_results)
        self._generate_comparison_charts(test_results)
        
        return test_results

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance tests - main method called by migration orchestrator"""
        logger.info("Starting comprehensive performance tests for migration validation")
        
        try:
            # Run all performance tests
            test_results = self.run_comprehensive_performance_tests()
            
            # Calculate overall success
            all_passed = all(result.success for result in test_results.values())
            
            # Generate summary
            summary = {
                'total_test_suites': len(test_results),
                'passed_test_suites': sum(1 for result in test_results.values() if result.success),
                'failed_test_suites': sum(1 for result in test_results.values() if not result.success),
                'overall_success': all_passed,
                'test_results': {name: result.to_dict() for name, result in test_results.items()}
            }
            
            logger.info(f"Comprehensive performance tests completed: {summary['passed_test_suites']}/{summary['total_test_suites']} passed")
            return summary
            
        except Exception as e:
            logger.error(f"Comprehensive performance tests failed: {e}")
            return {
                'total_test_suites': 0,
                'passed_test_suites': 0,
                'failed_test_suites': 0,
                'overall_success': False,
                'error': str(e)
            }

    def _save_test_results(self, test_results: Dict[str, PerformanceTestResult]):
        """Save test results to file"""
        results_data = {
            test_name: result.to_dict() 
            for test_name, result in test_results.items()
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.reports_dir / f"performance_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Performance test results saved: {results_file}")
    
    def _generate_performance_report(self, test_results: Dict[str, PerformanceTestResult]):
        """Generate comprehensive performance report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f"performance_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# PostgreSQL Migration Performance Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            sqlite_tests = {k: v for k, v in test_results.items() if k.startswith('sqlite')}
            postgresql_tests = {k: v for k, v in test_results.items() if k.startswith('postgresql')}
            
            f.write(f"- **SQLite Tests**: {len(sqlite_tests)} test suites\n")
            f.write(f"- **PostgreSQL Tests**: {len(postgresql_tests)} test suites\n")
            
            # Overall success rates
            sqlite_success = all(result.success for result in sqlite_tests.values())
            postgresql_success = all(result.success for result in postgresql_tests.values())
            
            f.write(f"- **SQLite Overall Success**: {'✅ PASS' if sqlite_success else '❌ FAIL'}\n")
            f.write(f"- **PostgreSQL Overall Success**: {'✅ PASS' if postgresql_success else '❌ FAIL'}\n\n")
            
            # Detailed Results
            for test_name, result in test_results.items():
                f.write(f"## {test_name.replace('_', ' ').title()}\n\n")
                f.write(f"- **Database**: {result.database_type}\n")
                f.write(f"- **Test Suite**: {result.test_suite}\n")
                f.write(f"- **Duration**: {result.total_duration_ms:.2f} ms\n")
                f.write(f"- **Success**: {'✅ PASS' if result.success else '❌ FAIL'}\n\n")
                
                f.write("### Summary Statistics\n\n")
                for key, value in result.summary_stats.items():
                    if isinstance(value, float):
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value:.2f}\n")
                    else:
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                
                f.write("\n")
            
            # Performance Comparison
            if sqlite_tests and postgresql_tests:
                f.write("## Performance Comparison\n\n")
                
                # Compare query performance
                sqlite_query = sqlite_tests.get('sqlite_query')
                postgresql_query = postgresql_tests.get('postgresql_query')
                
                if sqlite_query and postgresql_query:
                    sqlite_avg = sqlite_query.summary_stats.get('avg_query_time_ms', 0)
                    postgresql_avg = postgresql_query.summary_stats.get('avg_query_time_ms', 0)
                    
                    if sqlite_avg > 0:
                        improvement = ((sqlite_avg - postgresql_avg) / sqlite_avg) * 100
                        f.write(f"- **Query Performance**: PostgreSQL is {improvement:.1f}% {'faster' if improvement > 0 else 'slower'} than SQLite\n")
                
                # Compare throughput
                sqlite_qps = sqlite_query.summary_stats.get('queries_per_second', 0) if sqlite_query else 0
                postgresql_qps = postgresql_query.summary_stats.get('queries_per_second', 0) if postgresql_query else 0
                
                f.write(f"- **SQLite Throughput**: {sqlite_qps:.2f} queries/second\n")
                f.write(f"- **PostgreSQL Throughput**: {postgresql_qps:.2f} queries/second\n")
        
        logger.info(f"Performance report generated: {report_file}")
    
    def _generate_comparison_charts(self, test_results: Dict[str, PerformanceTestResult]):
        """Generate performance comparison charts"""
        try:
            # Response time comparison
            plt.figure(figsize=(12, 8))
            
            sqlite_data = []
            postgresql_data = []
            test_names = []
            
            for test_name, result in test_results.items():
                if 'query' in test_name:
                    avg_time = result.summary_stats.get('avg_query_time_ms', 0)
                    if 'sqlite' in test_name:
                        sqlite_data.append(avg_time)
                        test_names.append(test_name.replace('sqlite_', '').replace('_', ' ').title())
                    elif 'postgresql' in test_name:
                        postgresql_data.append(avg_time)
            
            if sqlite_data and postgresql_data:
                x = range(len(test_names))
                width = 0.35
                
                plt.bar([i - width/2 for i in x], sqlite_data, width, label='SQLite', alpha=0.8)
                plt.bar([i + width/2 for i in x], postgresql_data, width, label='PostgreSQL', alpha=0.8)
                
                plt.xlabel('Test Type')
                plt.ylabel('Average Response Time (ms)')
                plt.title('Database Performance Comparison')
                plt.xticks(x, test_names, rotation=45)
                plt.legend()
                plt.tight_layout()
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                chart_file = self.charts_dir / f"performance_comparison_{timestamp}.png"
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Performance comparison chart saved: {chart_file}")
        
        except Exception as e:
            logger.warning(f"Failed to generate charts: {e}")

def main():
    """Main performance testing operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PostgreSQL Migration Performance Tester")
    parser.add_argument("--test-type", choices=["connection", "query", "crud", "load", "all"],
                       default="all", help="Type of performance test to run")
    parser.add_argument("--database", choices=["sqlite", "postgresql", "both"],
                       default="both", help="Database to test")
    parser.add_argument("--config", help="Custom test configuration file")
    
    args = parser.parse_args()
    
    # Configuration
    source_db_uri = DatabaseConfig.SQLALCHEMY_DATABASE_URI
    target_db_uri = os.getenv('POSTGRES_URL', 'postgresql://hrp_user:password@localhost:5432/hrp_database')
    
    # Initialize performance tester
    tester = PerformanceTester(source_db_uri, target_db_uri)
    
    if args.test_type == "all":
        results = tester.run_comprehensive_performance_tests()
        
        # Print summary
        print("\n=== Performance Test Summary ===")
        for test_name, result in results.items():
            status = "PASS" if result.success else "FAIL"
            print(f"{test_name}: {status} ({result.total_duration_ms:.2f}ms)")
        
        return 0
    
    else:
        # Run specific test type
        databases = ["sqlite", "postgresql"] if args.database == "both" else [args.database]
        
        for db_type in databases:
            if args.test_type == "connection":
                result = tester.test_connection_performance(db_type)
            elif args.test_type == "query":
                result = tester.test_query_performance(db_type)
            elif args.test_type == "crud":
                result = tester.test_crud_performance(db_type)
            elif args.test_type == "load":
                result = tester.test_load_performance(db_type)
            
            print(f"\n{db_type} {args.test_type} test: {'PASS' if result.success else 'FAIL'}")
            print(f"Duration: {result.total_duration_ms:.2f}ms")
            print(f"Summary: {result.summary_stats}")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())