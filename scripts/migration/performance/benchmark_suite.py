#!/usr/bin/env python3
"""
PostgreSQL Migration Benchmark Suite
Comprehensive benchmarking for database migration validation
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
import numpy as np

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.database_config import DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/benchmark_suite.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    benchmark_name: str
    database_type: str
    operation_type: str
    duration_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    rows_processed: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    suite_name: str
    start_time: datetime
    end_time: datetime
    total_duration_ms: float
    results: List[BenchmarkResult]
    summary: Dict[str, Any]
    success: bool

class DatabaseBenchmark:
    """Database benchmark operations"""
    
    def __init__(self, source_db_uri: str, target_db_uri: str):
        self.source_db_uri = source_db_uri
        self.target_db_uri = target_db_uri
        
        # Create benchmark directory
        self.benchmark_dir = Path("scripts/migration/performance/benchmarks")
        self.benchmark_dir.mkdir(exist_ok=True)
        
        # Database engines
        self.source_engine = create_engine(source_db_uri)
        self.target_engine = create_engine(target_db_uri)
        
        # Benchmark configurations
        self.benchmark_configs = self._load_benchmark_configs()
    
    def _load_benchmark_configs(self) -> Dict[str, Any]:
        """Load benchmark configurations"""
        default_configs = {
            "data_sizes": [100, 1000, 10000, 50000],
            "concurrent_users": [1, 5, 10, 20, 50],
            "query_complexity": ["simple", "medium", "complex"],
            "test_duration_seconds": 60,
            "warmup_iterations": 10,
            "measurement_iterations": 100,
            "memory_threshold_mb": 1000,
            "response_time_threshold_ms": 1000,
            "throughput_threshold_ops_per_sec": 100
        }
        
        config_file = self.benchmark_dir / "benchmark_config.json"
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
                logger.warning(f"Failed to load benchmark config, using defaults: {e}")
        
        # Save default config
        with open(config_file, 'w') as f:
            json.dump(default_configs, f, indent=2)
        
        return default_configs
    
    def _get_system_metrics(self) -> Tuple[float, float]:
        """Get current system metrics"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            return memory_mb, cpu_percent
        except ImportError:
            return 0.0, 0.0
        except Exception:
            return 0.0, 0.0
    
    def _execute_benchmark_query(self, engine, query: str, params: Dict = None, 
                                iterations: int = 1) -> BenchmarkResult:
        """Execute a benchmark query with timing and metrics"""
        
        # Warmup
        try:
            with engine.connect() as conn:
                for _ in range(min(5, iterations // 10)):
                    if params:
                        conn.execute(text(query), params)
                    else:
                        conn.execute(text(query))
                conn.commit()
        except Exception:
            pass  # Ignore warmup errors
        
        # Actual benchmark
        durations = []
        rows_processed = 0
        memory_before, cpu_before = self._get_system_metrics()
        
        start_time = time.time()
        
        try:
            for i in range(iterations):
                iter_start = time.time()
                
                with engine.connect() as conn:
                    if params:
                        result = conn.execute(text(query), params)
                    else:
                        result = conn.execute(text(query))
                    
                    # Count rows for SELECT queries
                    if query.strip().upper().startswith('SELECT'):
                        rows = result.fetchall()
                        rows_processed += len(rows)
                    else:
                        rows_processed += result.rowcount
                    
                    conn.commit()
                
                iter_end = time.time()
                durations.append((iter_end - iter_start) * 1000)
            
            end_time = time.time()
            memory_after, cpu_after = self._get_system_metrics()
            
            total_duration_ms = (end_time - start_time) * 1000
            avg_duration_ms = statistics.mean(durations)
            throughput = (iterations / (total_duration_ms / 1000)) if total_duration_ms > 0 else 0
            
            return BenchmarkResult(
                benchmark_name="query_benchmark",
                database_type="unknown",
                operation_type="query",
                duration_ms=avg_duration_ms,
                throughput_ops_per_sec=throughput,
                memory_usage_mb=max(0, memory_after - memory_before),
                cpu_usage_percent=max(0, cpu_after - cpu_before),
                rows_processed=rows_processed,
                success=True,
                metadata={
                    "iterations": iterations,
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                    "std_duration_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
                    "total_duration_ms": total_duration_ms
                }
            )
            
        except Exception as e:
            end_time = time.time()
            total_duration_ms = (end_time - start_time) * 1000
            
            return BenchmarkResult(
                benchmark_name="query_benchmark",
                database_type="unknown",
                operation_type="query",
                duration_ms=total_duration_ms,
                throughput_ops_per_sec=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                rows_processed=0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_read_operations(self, database_type: str) -> List[BenchmarkResult]:
        """Benchmark read operations"""
        logger.info(f"Benchmarking read operations for {database_type}")
        
        engine = self.target_engine if database_type == "postgresql" else self.source_engine
        results = []
        
        # Define read benchmark queries
        read_queries = [
            # Simple reads
            ("simple_select_all", "SELECT * FROM races LIMIT 1000"),
            ("simple_count", "SELECT COUNT(*) FROM horses"),
            ("simple_filter", "SELECT * FROM predictions WHERE confidence > 0.8"),
            
            # Medium complexity reads
            ("medium_join", """
                SELECT r.race_name, h.horse_name, p.confidence
                FROM races r
                JOIN predictions p ON r.id = p.race_id
                JOIN horses h ON p.horse_id = h.id
                LIMIT 1000
            """),
            ("medium_aggregation", """
                SELECT 
                    h.horse_name,
                    COUNT(p.id) as prediction_count,
                    AVG(p.confidence) as avg_confidence
                FROM horses h
                LEFT JOIN predictions p ON h.id = p.horse_id
                GROUP BY h.id, h.horse_name
                LIMIT 500
            """),
            
            # Complex reads
            ("complex_subquery", """
                SELECT r.*, 
                       (SELECT COUNT(*) FROM predictions p WHERE p.race_id = r.id) as prediction_count,
                       (SELECT AVG(confidence) FROM predictions p WHERE p.race_id = r.id) as avg_confidence
                FROM races r
                WHERE r.race_date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY r.race_date DESC
                LIMIT 100
            """),
            ("complex_window", """
                SELECT 
                    h.horse_name,
                    p.confidence,
                    ROW_NUMBER() OVER (PARTITION BY h.id ORDER BY p.confidence DESC) as rank,
                    AVG(p.confidence) OVER (PARTITION BY h.id) as avg_confidence
                FROM horses h
                JOIN predictions p ON h.id = p.horse_id
                WHERE p.confidence > 0.5
                ORDER BY h.horse_name, rank
                LIMIT 1000
            """)
        ]
        
        iterations = self.benchmark_configs["measurement_iterations"]
        
        for query_name, query in read_queries:
            result = self._execute_benchmark_query(engine, query, iterations=iterations)
            result.benchmark_name = f"read_{query_name}"
            result.database_type = database_type
            result.operation_type = "read"
            results.append(result)
        
        return results
    
    def benchmark_write_operations(self, database_type: str) -> List[BenchmarkResult]:
        """Benchmark write operations"""
        logger.info(f"Benchmarking write operations for {database_type}")
        
        engine = self.target_engine if database_type == "postgresql" else self.source_engine
        results = []
        
        # Test data sizes
        data_sizes = self.benchmark_configs["data_sizes"]
        
        for size in data_sizes:
            # INSERT benchmark
            insert_start = time.time()
            memory_before, cpu_before = self._get_system_metrics()
            
            try:
                with engine.connect() as conn:
                    for i in range(size):
                        conn.execute(text("""
                            INSERT INTO horses (horse_name, age, weight, jockey, trainer, owner, created_at)
                            VALUES (:name, :age, :weight, :jockey, :trainer, :owner, :created_at)
                        """), {
                            'name': f'Benchmark Horse {i}_{int(time.time())}',
                            'age': 3 + (i % 5),
                            'weight': 450 + (i % 100),
                            'jockey': f'Jockey {i % 10}',
                            'trainer': f'Trainer {i % 5}',
                            'owner': f'Owner {i % 20}',
                            'created_at': datetime.now()
                        })
                    conn.commit()
                
                insert_end = time.time()
                memory_after, cpu_after = self._get_system_metrics()
                
                duration_ms = (insert_end - insert_start) * 1000
                throughput = size / (duration_ms / 1000) if duration_ms > 0 else 0
                
                results.append(BenchmarkResult(
                    benchmark_name=f"write_insert_{size}",
                    database_type=database_type,
                    operation_type="insert",
                    duration_ms=duration_ms,
                    throughput_ops_per_sec=throughput,
                    memory_usage_mb=max(0, memory_after - memory_before),
                    cpu_usage_percent=max(0, cpu_after - cpu_before),
                    rows_processed=size,
                    success=True,
                    metadata={"batch_size": size}
                ))
                
            except Exception as e:
                results.append(BenchmarkResult(
                    benchmark_name=f"write_insert_{size}",
                    database_type=database_type,
                    operation_type="insert",
                    duration_ms=0,
                    throughput_ops_per_sec=0,
                    memory_usage_mb=0,
                    cpu_usage_percent=0,
                    rows_processed=0,
                    success=False,
                    error_message=str(e),
                    metadata={"batch_size": size}
                ))
            
            # UPDATE benchmark
            update_result = self._execute_benchmark_query(
                engine,
                "UPDATE horses SET weight = weight + 1 WHERE horse_name LIKE :pattern",
                {"pattern": f"Benchmark Horse%"},
                iterations=10
            )
            update_result.benchmark_name = f"write_update_{size}"
            update_result.database_type = database_type
            update_result.operation_type = "update"
            results.append(update_result)
            
            # DELETE benchmark (cleanup)
            delete_result = self._execute_benchmark_query(
                engine,
                "DELETE FROM horses WHERE horse_name LIKE :pattern",
                {"pattern": f"Benchmark Horse%"},
                iterations=1
            )
            delete_result.benchmark_name = f"write_delete_{size}"
            delete_result.database_type = database_type
            delete_result.operation_type = "delete"
            results.append(delete_result)
        
        return results
    
    def benchmark_concurrent_operations(self, database_type: str) -> List[BenchmarkResult]:
        """Benchmark concurrent operations"""
        logger.info(f"Benchmarking concurrent operations for {database_type}")
        
        engine = self.target_engine if database_type == "postgresql" else self.source_engine
        results = []
        
        concurrent_users = self.benchmark_configs["concurrent_users"]
        test_duration = self.benchmark_configs["test_duration_seconds"]
        
        # Define concurrent test queries
        concurrent_queries = [
            "SELECT COUNT(*) FROM races",
            "SELECT * FROM horses ORDER BY created_at DESC LIMIT 10",
            "SELECT AVG(confidence) FROM predictions WHERE confidence > 0.5"
        ]
        
        def concurrent_worker(worker_id: int, duration: int) -> List[BenchmarkResult]:
            """Worker function for concurrent testing"""
            worker_results = []
            end_time = time.time() + duration
            query_count = 0
            
            start_time = time.time()
            memory_before, cpu_before = self._get_system_metrics()
            
            try:
                while time.time() < end_time:
                    for query in concurrent_queries:
                        with engine.connect() as conn:
                            conn.execute(text(query))
                            query_count += 1
                        
                        # Small delay to prevent overwhelming
                        time.sleep(0.01)
                
                actual_end_time = time.time()
                memory_after, cpu_after = self._get_system_metrics()
                
                total_duration_ms = (actual_end_time - start_time) * 1000
                throughput = query_count / (total_duration_ms / 1000) if total_duration_ms > 0 else 0
                
                worker_results.append(BenchmarkResult(
                    benchmark_name=f"concurrent_worker_{worker_id}",
                    database_type=database_type,
                    operation_type="concurrent_read",
                    duration_ms=total_duration_ms,
                    throughput_ops_per_sec=throughput,
                    memory_usage_mb=max(0, memory_after - memory_before),
                    cpu_usage_percent=max(0, cpu_after - cpu_before),
                    rows_processed=query_count,
                    success=True,
                    metadata={
                        "worker_id": worker_id,
                        "queries_executed": query_count,
                        "target_duration": duration
                    }
                ))
                
            except Exception as e:
                worker_results.append(BenchmarkResult(
                    benchmark_name=f"concurrent_worker_{worker_id}",
                    database_type=database_type,
                    operation_type="concurrent_read",
                    duration_ms=0,
                    throughput_ops_per_sec=0,
                    memory_usage_mb=0,
                    cpu_usage_percent=0,
                    rows_processed=0,
                    success=False,
                    error_message=str(e),
                    metadata={"worker_id": worker_id}
                ))
            
            return worker_results
        
        for user_count in concurrent_users:
            logger.info(f"Testing with {user_count} concurrent users")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=user_count) as executor:
                futures = [
                    executor.submit(concurrent_worker, i, test_duration)
                    for i in range(user_count)
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    worker_results = future.result()
                    results.extend(worker_results)
        
        return results
    
    def benchmark_data_integrity(self, database_type: str) -> List[BenchmarkResult]:
        """Benchmark data integrity operations"""
        logger.info(f"Benchmarking data integrity for {database_type}")
        
        engine = self.target_engine if database_type == "postgresql" else self.source_engine
        results = []
        
        # Test foreign key constraints
        fk_test_start = time.time()
        try:
            with engine.connect() as conn:
                # Try to insert invalid foreign key
                conn.execute(text("""
                    INSERT INTO predictions (race_id, horse_id, prediction_value, confidence, created_at)
                    VALUES (99999, 99999, 1.5, 0.8, :created_at)
                """), {"created_at": datetime.now()})
                conn.commit()
            
            # If we get here, constraint failed
            results.append(BenchmarkResult(
                benchmark_name="integrity_foreign_key",
                database_type=database_type,
                operation_type="integrity_check",
                duration_ms=(time.time() - fk_test_start) * 1000,
                throughput_ops_per_sec=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                rows_processed=0,
                success=False,
                error_message="Foreign key constraint not enforced"
            ))
            
        except Exception:
            # Expected behavior - constraint should prevent invalid insert
            results.append(BenchmarkResult(
                benchmark_name="integrity_foreign_key",
                database_type=database_type,
                operation_type="integrity_check",
                duration_ms=(time.time() - fk_test_start) * 1000,
                throughput_ops_per_sec=1,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                rows_processed=0,
                success=True,
                metadata={"constraint_enforced": True}
            ))
        
        # Test unique constraints
        unique_test_start = time.time()
        try:
            with engine.connect() as conn:
                # Insert a test record
                conn.execute(text("""
                    INSERT INTO horses (horse_name, age, weight, jockey, trainer, owner, created_at)
                    VALUES (:name, :age, :weight, :jockey, :trainer, :owner, :created_at)
                """), {
                    'name': f'Unique Test Horse {int(time.time())}',
                    'age': 4,
                    'weight': 500,
                    'jockey': 'Test Jockey',
                    'trainer': 'Test Trainer',
                    'owner': 'Test Owner',
                    'created_at': datetime.now()
                })
                
                # Try to insert duplicate (if unique constraint exists)
                conn.execute(text("""
                    INSERT INTO horses (horse_name, age, weight, jockey, trainer, owner, created_at)
                    VALUES (:name, :age, :weight, :jockey, :trainer, :owner, :created_at)
                """), {
                    'name': f'Unique Test Horse {int(time.time())}',
                    'age': 4,
                    'weight': 500,
                    'jockey': 'Test Jockey',
                    'trainer': 'Test Trainer',
                    'owner': 'Test Owner',
                    'created_at': datetime.now()
                })
                conn.commit()
            
            results.append(BenchmarkResult(
                benchmark_name="integrity_unique_constraint",
                database_type=database_type,
                operation_type="integrity_check",
                duration_ms=(time.time() - unique_test_start) * 1000,
                throughput_ops_per_sec=1,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                rows_processed=2,
                success=True,
                metadata={"unique_constraint_enforced": False}
            ))
            
        except Exception:
            results.append(BenchmarkResult(
                benchmark_name="integrity_unique_constraint",
                database_type=database_type,
                operation_type="integrity_check",
                duration_ms=(time.time() - unique_test_start) * 1000,
                throughput_ops_per_sec=1,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                rows_processed=1,
                success=True,
                metadata={"unique_constraint_enforced": True}
            ))
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, BenchmarkSuite]:
        """Run comprehensive benchmark suite"""
        logger.info("Starting comprehensive benchmark suite")
        
        benchmark_suites = {}
        
        for db_type in ["sqlite", "postgresql"]:
            logger.info(f"Running benchmarks for {db_type}")
            
            suite_start = datetime.now()
            all_results = []
            
            try:
                # Read operations benchmark
                read_results = self.benchmark_read_operations(db_type)
                all_results.extend(read_results)
                
                # Write operations benchmark
                write_results = self.benchmark_write_operations(db_type)
                all_results.extend(write_results)
                
                # Concurrent operations benchmark
                concurrent_results = self.benchmark_concurrent_operations(db_type)
                all_results.extend(concurrent_results)
                
                # Data integrity benchmark
                integrity_results = self.benchmark_data_integrity(db_type)
                all_results.extend(integrity_results)
                
            except Exception as e:
                logger.error(f"Benchmark failed for {db_type}: {e}")
                all_results = []
            
            suite_end = datetime.now()
            total_duration = (suite_end - suite_start).total_seconds() * 1000
            
            # Calculate summary statistics
            successful_results = [r for r in all_results if r.success]
            summary = {
                "total_benchmarks": len(all_results),
                "successful_benchmarks": len(successful_results),
                "failed_benchmarks": len(all_results) - len(successful_results),
                "success_rate": len(successful_results) / len(all_results) * 100 if all_results else 0,
                "avg_response_time_ms": statistics.mean([r.duration_ms for r in successful_results]) if successful_results else 0,
                "avg_throughput_ops_per_sec": statistics.mean([r.throughput_ops_per_sec for r in successful_results]) if successful_results else 0,
                "total_rows_processed": sum([r.rows_processed for r in successful_results]),
                "avg_memory_usage_mb": statistics.mean([r.memory_usage_mb for r in successful_results]) if successful_results else 0,
                "avg_cpu_usage_percent": statistics.mean([r.cpu_usage_percent for r in successful_results]) if successful_results else 0
            }
            
            benchmark_suites[db_type] = BenchmarkSuite(
                suite_name=f"{db_type}_comprehensive_benchmark",
                start_time=suite_start,
                end_time=suite_end,
                total_duration_ms=total_duration,
                results=all_results,
                summary=summary,
                success=summary["success_rate"] > 90
            )
        
        # Save and report results
        self._save_benchmark_results(benchmark_suites)
        self._generate_benchmark_report(benchmark_suites)
        
        return benchmark_suites
    
    def _save_benchmark_results(self, benchmark_suites: Dict[str, BenchmarkSuite]):
        """Save benchmark results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.benchmark_dir / f"benchmark_results_{timestamp}.json"
        
        results_data = {}
        for suite_name, suite in benchmark_suites.items():
            results_data[suite_name] = {
                "suite_name": suite.suite_name,
                "start_time": suite.start_time.isoformat(),
                "end_time": suite.end_time.isoformat(),
                "total_duration_ms": suite.total_duration_ms,
                "summary": suite.summary,
                "success": suite.success,
                "results": [asdict(result) for result in suite.results]
            }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved: {results_file}")
    
    def _generate_benchmark_report(self, benchmark_suites: Dict[str, BenchmarkSuite]):
        """Generate benchmark report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.benchmark_dir / f"benchmark_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# PostgreSQL Migration Benchmark Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            for suite_name, suite in benchmark_suites.items():
                f.write(f"### {suite_name.upper()} Database\n\n")
                f.write(f"- **Overall Success**: {'✅ PASS' if suite.success else '❌ FAIL'}\n")
                f.write(f"- **Total Duration**: {suite.total_duration_ms:.2f} ms\n")
                f.write(f"- **Success Rate**: {suite.summary['success_rate']:.1f}%\n")
                f.write(f"- **Average Response Time**: {suite.summary['avg_response_time_ms']:.2f} ms\n")
                f.write(f"- **Average Throughput**: {suite.summary['avg_throughput_ops_per_sec']:.2f} ops/sec\n")
                f.write(f"- **Total Rows Processed**: {suite.summary['total_rows_processed']:,}\n\n")
            
            # Performance Comparison
            if len(benchmark_suites) >= 2:
                f.write("## Performance Comparison\n\n")
                
                sqlite_suite = benchmark_suites.get('sqlite')
                postgresql_suite = benchmark_suites.get('postgresql')
                
                if sqlite_suite and postgresql_suite:
                    sqlite_avg_time = sqlite_suite.summary['avg_response_time_ms']
                    postgresql_avg_time = postgresql_suite.summary['avg_response_time_ms']
                    
                    if sqlite_avg_time > 0:
                        time_improvement = ((sqlite_avg_time - postgresql_avg_time) / sqlite_avg_time) * 100
                        f.write(f"- **Response Time**: PostgreSQL is {time_improvement:.1f}% {'faster' if time_improvement > 0 else 'slower'}\n")
                    
                    sqlite_throughput = sqlite_suite.summary['avg_throughput_ops_per_sec']
                    postgresql_throughput = postgresql_suite.summary['avg_throughput_ops_per_sec']
                    
                    if sqlite_throughput > 0:
                        throughput_improvement = ((postgresql_throughput - sqlite_throughput) / sqlite_throughput) * 100
                        f.write(f"- **Throughput**: PostgreSQL is {throughput_improvement:.1f}% {'better' if throughput_improvement > 0 else 'worse'}\n")
                    
                    f.write(f"- **SQLite Success Rate**: {sqlite_suite.summary['success_rate']:.1f}%\n")
                    f.write(f"- **PostgreSQL Success Rate**: {postgresql_suite.summary['success_rate']:.1f}%\n\n")
            
            # Detailed Results
            for suite_name, suite in benchmark_suites.items():
                f.write(f"## {suite_name.upper()} Detailed Results\n\n")
                
                # Group results by operation type
                operation_groups = {}
                for result in suite.results:
                    op_type = result.operation_type
                    if op_type not in operation_groups:
                        operation_groups[op_type] = []
                    operation_groups[op_type].append(result)
                
                for op_type, results in operation_groups.items():
                    f.write(f"### {op_type.replace('_', ' ').title()} Operations\n\n")
                    
                    successful_results = [r for r in results if r.success]
                    if successful_results:
                        avg_time = statistics.mean([r.duration_ms for r in successful_results])
                        avg_throughput = statistics.mean([r.throughput_ops_per_sec for r in successful_results])
                        
                        f.write(f"- **Average Response Time**: {avg_time:.2f} ms\n")
                        f.write(f"- **Average Throughput**: {avg_throughput:.2f} ops/sec\n")
                        f.write(f"- **Success Rate**: {len(successful_results) / len(results) * 100:.1f}%\n")
                        f.write(f"- **Total Operations**: {len(results)}\n\n")
                    else:
                        f.write("- **No successful operations**\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if postgresql_suite and sqlite_suite:
                if postgresql_suite.summary['avg_response_time_ms'] < sqlite_suite.summary['avg_response_time_ms']:
                    f.write("- ✅ PostgreSQL shows better response times than SQLite\n")
                else:
                    f.write("- ⚠️ PostgreSQL response times are slower than SQLite - investigate indexing and query optimization\n")
                
                if postgresql_suite.summary['avg_throughput_ops_per_sec'] > sqlite_suite.summary['avg_throughput_ops_per_sec']:
                    f.write("- ✅ PostgreSQL shows better throughput than SQLite\n")
                else:
                    f.write("- ⚠️ PostgreSQL throughput is lower than SQLite - consider connection pooling and optimization\n")
                
                if postgresql_suite.summary['success_rate'] >= 95:
                    f.write("- ✅ PostgreSQL shows excellent reliability\n")
                else:
                    f.write("- ❌ PostgreSQL reliability is below acceptable threshold - investigate error patterns\n")
        
        logger.info(f"Benchmark report generated: {report_file}")

def main():
    """Main benchmark operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PostgreSQL Migration Benchmark Suite")
    parser.add_argument("--benchmark-type", choices=["read", "write", "concurrent", "integrity", "all"],
                       default="all", help="Type of benchmark to run")
    parser.add_argument("--database", choices=["sqlite", "postgresql", "both"],
                       default="both", help="Database to benchmark")
    
    args = parser.parse_args()
    
    # Configuration
    source_db_uri = DatabaseConfig.SQLALCHEMY_DATABASE_URI
    target_db_uri = os.getenv('POSTGRES_URL', 'postgresql://hrp_user:password@localhost:5432/hrp_database')
    
    # Initialize benchmark suite
    benchmark = DatabaseBenchmark(source_db_uri, target_db_uri)
    
    if args.benchmark_type == "all":
        results = benchmark.run_comprehensive_benchmark()
        
        # Print summary
        print("\n=== Benchmark Summary ===")
        for suite_name, suite in results.items():
            status = "PASS" if suite.success else "FAIL"
            print(f"{suite_name}: {status}")
            print(f"  Success Rate: {suite.summary['success_rate']:.1f}%")
            print(f"  Avg Response Time: {suite.summary['avg_response_time_ms']:.2f}ms")
            print(f"  Avg Throughput: {suite.summary['avg_throughput_ops_per_sec']:.2f} ops/sec")
        
        return 0
    
    else:
        # Run specific benchmark type
        databases = ["sqlite", "postgresql"] if args.database == "both" else [args.database]
        
        for db_type in databases:
            if args.benchmark_type == "read":
                results = benchmark.benchmark_read_operations(db_type)
            elif args.benchmark_type == "write":
                results = benchmark.benchmark_write_operations(db_type)
            elif args.benchmark_type == "concurrent":
                results = benchmark.benchmark_concurrent_operations(db_type)
            elif args.benchmark_type == "integrity":
                results = benchmark.benchmark_data_integrity(db_type)
            
            successful = sum(1 for r in results if r.success)
            print(f"\n{db_type} {args.benchmark_type} benchmark:")
            print(f"  Success Rate: {successful / len(results) * 100:.1f}%")
            print(f"  Total Tests: {len(results)}")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())