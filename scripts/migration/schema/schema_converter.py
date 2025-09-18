#!/usr/bin/env python3
"""
PostgreSQL Schema Converter
Converts SQLAlchemy models to PostgreSQL-optimized schema with data type mapping
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.types import (
    Integer, String, Float, Boolean, DateTime, Text, 
    DECIMAL, VARCHAR, TIMESTAMP
)

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.database_config import DatabaseConfig
from models.sqlalchemy_models import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/schema_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PostgreSQLSchemaConverter:
    """Converts SQLAlchemy schema to PostgreSQL-optimized schema"""
    
    def __init__(self, source_db_uri: str, target_db_uri: str):
        self.source_db_uri = source_db_uri
        self.target_db_uri = target_db_uri
        self.source_engine = create_engine(source_db_uri)
        self.target_engine = create_engine(target_db_uri)
        self.metadata = MetaData()
        
        # PostgreSQL type mappings
        self.type_mappings = {
            'INTEGER': 'INTEGER',
            'VARCHAR': 'VARCHAR',
            'TEXT': 'TEXT',
            'FLOAT': 'DECIMAL',
            'BOOLEAN': 'BOOLEAN',
            'DATETIME': 'TIMESTAMPTZ',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'DECIMAL': 'DECIMAL',
            'JSON': 'JSONB'
        }
        
        # PostgreSQL-specific optimizations
        self.pg_optimizations = {
            'horses': {
                'indexes': [
                    'CREATE INDEX idx_horses_name_gin ON horses USING gin(name gin_trgm_ops);',
                    'CREATE INDEX idx_horses_trainer ON horses(trainer);',
                    'CREATE INDEX idx_horses_jockey ON horses(jockey);',
                    'CREATE INDEX idx_horses_rating ON horses(rating);'
                ],
                'constraints': [
                    'ALTER TABLE horses ADD CONSTRAINT chk_age CHECK (age > 0 AND age <= 30);',
                    'ALTER TABLE horses ADD CONSTRAINT chk_wins_places CHECK (wins <= places);',
                    'ALTER TABLE horses ADD CONSTRAINT chk_places_runs CHECK (places <= runs);'
                ]
            },
            'races': {
                'indexes': [
                    'CREATE INDEX idx_races_date ON races(date);',
                    'CREATE INDEX idx_races_track ON races(track);',
                    'CREATE INDEX idx_races_status ON races(status);',
                    'CREATE INDEX idx_races_date_status ON races(date, status);'
                ],
                'constraints': [
                    'ALTER TABLE races ADD CONSTRAINT chk_distance CHECK (distance > 0);',
                    'ALTER TABLE races ADD CONSTRAINT chk_prize_money CHECK (prize_money >= 0);'
                ]
            },
            'predictions': {
                'indexes': [
                    'CREATE INDEX idx_predictions_race_id ON predictions(race_id);',
                    'CREATE INDEX idx_predictions_horse_id ON predictions(horse_id);',
                    'CREATE INDEX idx_predictions_confidence ON predictions(confidence);'
                ],
                'constraints': [
                    'ALTER TABLE predictions ADD CONSTRAINT chk_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0);',
                    'ALTER TABLE predictions ADD CONSTRAINT chk_predicted_position CHECK (predicted_position > 0);'
                ]
            }
        }
    
    def analyze_source_schema(self) -> Dict[str, Any]:
        """Analyze the source database schema"""
        logger.info("Analyzing source database schema...")
        
        try:
            inspector = inspect(self.source_engine)
            schema_info = {
                'tables': {},
                'indexes': {},
                'foreign_keys': {},
                'constraints': {}
            }
            
            # Get table information
            for table_name in inspector.get_table_names():
                logger.info(f"Analyzing table: {table_name}")
                
                # Get columns
                columns = inspector.get_columns(table_name)
                schema_info['tables'][table_name] = {
                    'columns': columns,
                    'row_count': self._get_table_row_count(table_name)
                }
                
                # Get indexes
                indexes = inspector.get_indexes(table_name)
                schema_info['indexes'][table_name] = indexes
                
                # Get foreign keys
                foreign_keys = inspector.get_foreign_keys(table_name)
                schema_info['foreign_keys'][table_name] = foreign_keys
                
                # Get check constraints
                try:
                    constraints = inspector.get_check_constraints(table_name)
                    schema_info['constraints'][table_name] = constraints
                except:
                    schema_info['constraints'][table_name] = []
            
            logger.info(f"Schema analysis complete. Found {len(schema_info['tables'])} tables")
            return schema_info
            
        except Exception as e:
            logger.error(f"Error analyzing source schema: {e}")
            raise
    
    def _get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table"""
        try:
            with self.source_engine.connect() as conn:
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                return result.scalar()
        except:
            return 0
    
    def convert_column_type(self, column_info: Dict[str, Any]) -> str:
        """Convert SQLAlchemy column type to PostgreSQL type"""
        column_type = str(column_info['type']).upper()
        
        # Handle specific type conversions
        if 'VARCHAR' in column_type:
            if '(' in column_type:
                return column_type  # Keep length specification
            else:
                return 'VARCHAR(255)'  # Default length
        elif 'INTEGER' in column_type:
            return 'INTEGER'
        elif 'FLOAT' in column_type or 'REAL' in column_type:
            return 'DECIMAL(12,2)'  # More precise for financial data
        elif 'BOOLEAN' in column_type:
            return 'BOOLEAN'
        elif 'DATETIME' in column_type or 'TIMESTAMP' in column_type:
            return 'TIMESTAMPTZ'
        elif 'TEXT' in column_type:
            return 'TEXT'
        elif 'DECIMAL' in column_type:
            return column_type
        else:
            # Default fallback
            return 'TEXT'
    
    def generate_postgresql_schema(self, schema_info: Dict[str, Any]) -> str:
        """Generate PostgreSQL schema DDL"""
        logger.info("Generating PostgreSQL schema...")
        
        ddl_statements = []
        
        # Add header
        ddl_statements.append("-- PostgreSQL Schema Generated from SQLAlchemy Models")
        ddl_statements.append(f"-- Generated on: {datetime.now().isoformat()}")
        ddl_statements.append("")
        
        # Add extensions
        ddl_statements.extend([
            "-- Enable required extensions",
            'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";',
            'CREATE EXTENSION IF NOT EXISTS "pg_trgm";',
            'CREATE EXTENSION IF NOT EXISTS "btree_gin";',
            ""
        ])
        
        # Add custom types
        ddl_statements.extend([
            "-- Create custom types",
            "CREATE TYPE race_status AS ENUM ('upcoming', 'running', 'completed', 'cancelled');",
            "CREATE TYPE horse_sex AS ENUM ('M', 'F', 'G', 'C');",
            "CREATE TYPE surface_type AS ENUM ('dirt', 'turf', 'synthetic', 'all_weather');",
            "CREATE TYPE track_condition AS ENUM ('fast', 'good', 'firm', 'soft', 'heavy', 'sloppy', 'muddy', 'frozen');",
            ""
        ])
        
        # Check if source database has any tables
        if not schema_info['tables']:
            logger.warning("Source database has no tables. Generating minimal schema.")
            ddl_statements.append("-- No tables found in source database")
            ddl_statements.append("-- Schema conversion completed with empty source")
            return "\n".join(ddl_statements)

        # Generate table creation statements
        for table_name, table_info in schema_info['tables'].items():
            ddl_statements.append(f"-- Table: {table_name}")
            ddl_statements.append(f"CREATE TABLE {table_name} (")
            
            column_definitions = []
            for column in table_info['columns']:
                col_def = self._generate_column_definition(column, table_name)
                column_definitions.append(f"    {col_def}")
            
            ddl_statements.append(",\n".join(column_definitions))
            ddl_statements.append(");")
            ddl_statements.append("")

        # Add foreign key constraints
        for table_name, foreign_keys in schema_info['foreign_keys'].items():
            if foreign_keys:
                ddl_statements.append(f"-- Foreign keys for {table_name}")
                for fk in foreign_keys:
                    fk_statement = self._generate_foreign_key_statement(table_name, fk)
                    ddl_statements.append(fk_statement)
                ddl_statements.append("")

        # Add indexes - only for tables that exist in source
        ddl_statements.append("-- Indexes for performance optimization")
        for table_name in schema_info['tables'].keys():
            if table_name in self.pg_optimizations:
                for index_sql in self.pg_optimizations[table_name].get('indexes', []):
                    ddl_statements.append(index_sql)
        ddl_statements.append("")

        # Add constraints - only for tables that exist in source
        ddl_statements.append("-- Additional constraints")
        for table_name in schema_info['tables'].keys():
            if table_name in self.pg_optimizations:
                for constraint_sql in self.pg_optimizations[table_name].get('constraints', []):
                    ddl_statements.append(constraint_sql)
        ddl_statements.append("")
        
        # Add triggers and functions
        ddl_statements.extend(self._generate_triggers_and_functions(schema_info))
        
        return "\n".join(ddl_statements)
    
    def _generate_column_definition(self, column: Dict[str, Any], table_name: str) -> str:
        """Generate PostgreSQL column definition"""
        col_name = column['name']
        col_type = self.convert_column_type(column)
        
        # Handle special cases
        if col_name == 'id' and table_name != 'race_horses':
            col_def = f"{col_name} SERIAL PRIMARY KEY"
        else:
            col_def = f"{col_name} {col_type}"
            
            # Add constraints
            if not column.get('nullable', True):
                col_def += " NOT NULL"
            
            if column.get('default') is not None:
                default_val = column['default']
                if isinstance(default_val, str):
                    col_def += f" DEFAULT '{default_val}'"
                else:
                    col_def += f" DEFAULT {default_val}"
            
            # Add unique constraint
            if column.get('unique', False):
                col_def += " UNIQUE"
        
        return col_def
    
    def _generate_foreign_key_statement(self, table_name: str, fk_info: Dict[str, Any]) -> str:
        """Generate foreign key constraint statement"""
        constraint_name = fk_info.get('name', f"fk_{table_name}_{fk_info['constrained_columns'][0]}")
        local_column = fk_info['constrained_columns'][0]
        referenced_table = fk_info['referred_table']
        referenced_column = fk_info['referred_columns'][0]
        
        return (f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} "
                f"FOREIGN KEY ({local_column}) REFERENCES {referenced_table}({referenced_column}) "
                f"ON DELETE CASCADE;")
    
    def _generate_triggers_and_functions(self, schema_info: Dict[str, Any]) -> List[str]:
        """Generate PostgreSQL triggers and functions"""
        statements = [
            "-- Automatic timestamp update function",
            "CREATE OR REPLACE FUNCTION update_updated_at_column()",
            "RETURNS TRIGGER AS $$",
            "BEGIN",
            "    NEW.updated_at = CURRENT_TIMESTAMP;",
            "    RETURN NEW;",
            "END;",
            "$$ language 'plpgsql';",
            ""
        ]
        
        # Only generate triggers for tables that exist in source and have updated_at column
        if schema_info['tables']:
            statements.append("-- Triggers for automatic timestamp updates")
            for table_name in schema_info['tables'].keys():
                # Check if table has updated_at column
                table_info = schema_info['tables'][table_name]
                has_updated_at = any(col['name'] == 'updated_at' for col in table_info['columns'])
                if has_updated_at:
                    statements.extend([
                        f"CREATE TRIGGER update_{table_name}_updated_at BEFORE UPDATE ON {table_name}",
                        "    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
                        ""
                    ])
        
        return statements
    
    def validate_conversion(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the schema conversion"""
        logger.info("Validating schema conversion...")
        
        validation_results = {
            'tables_converted': len(schema_info['tables']),
            'total_columns': sum(len(table['columns']) for table in schema_info['tables'].values()),
            'foreign_keys': sum(len(fks) for fks in schema_info['foreign_keys'].values()),
            'indexes_created': 0,
            'constraints_added': 0,
            'issues': []
        }
        
        # Count optimizations
        for table_name in schema_info['tables'].keys():
            if table_name in self.pg_optimizations:
                validation_results['indexes_created'] += len(
                    self.pg_optimizations[table_name].get('indexes', [])
                )
                validation_results['constraints_added'] += len(
                    self.pg_optimizations[table_name].get('constraints', [])
                )
        
        # Check for potential issues
        for table_name, table_info in schema_info['tables'].items():
            # Check for large text columns without indexes
            for column in table_info['columns']:
                if 'TEXT' in str(column['type']).upper() and table_info['row_count'] > 1000:
                    validation_results['issues'].append(
                        f"Table {table_name} has TEXT column {column['name']} with {table_info['row_count']} rows - consider adding full-text search index"
                    )
        
        logger.info(f"Validation complete: {validation_results}")
        return validation_results

    def create_indexes_and_constraints(self, target_db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create indexes and constraints on the target database"""
        logger.info("Creating indexes and constraints...")
        
        try:
            results = {
                'indexes_created': 0,
                'constraints_added': 0,
                'errors': [],
                'success': True
            }
            
            with self.target_engine.connect() as conn:
                with conn.begin():
                    # Get list of existing tables
                    inspector = inspect(self.target_engine)
                    existing_tables = inspector.get_table_names()
                    
                    # Create indexes for existing tables
                    for table_name in existing_tables:
                        if table_name in self.pg_optimizations:
                            # Create indexes
                            for index_sql in self.pg_optimizations[table_name].get('indexes', []):
                                try:
                                    from sqlalchemy import text
                                    conn.execute(text(index_sql))
                                    results['indexes_created'] += 1
                                    logger.info(f"Created index: {index_sql}")
                                except Exception as e:
                                    error_msg = f"Failed to create index: {index_sql} - {e}"
                                    logger.warning(error_msg)
                                    results['errors'].append(error_msg)
                            
                            # Add constraints
                            for constraint_sql in self.pg_optimizations[table_name].get('constraints', []):
                                try:
                                    from sqlalchemy import text
                                    conn.execute(text(constraint_sql))
                                    results['constraints_added'] += 1
                                    logger.info(f"Added constraint: {constraint_sql}")
                                except Exception as e:
                                    error_msg = f"Failed to add constraint: {constraint_sql} - {e}"
                                    logger.warning(error_msg)
                                    results['errors'].append(error_msg)
            
            if results['errors']:
                results['success'] = len(results['errors']) < (results['indexes_created'] + results['constraints_added']) / 2
            
            logger.info(f"Index and constraint creation completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to create indexes and constraints: {e}")
            return {
                'indexes_created': 0,
                'constraints_added': 0,
                'errors': [str(e)],
                'success': False
            }
    
    def _parse_ddl_statements(self, schema_ddl: str) -> List[str]:
        """
        Parse DDL into individual statements, properly handling:
        - Comments (-- and /* */)
        - Dollar-quoted strings ($$)
        - Multi-line statements
        """
        statements = []
        current_statement = []
        in_dollar_quote = False
        dollar_tag = None
        
        lines = schema_ddl.split('\n')
        
        for line in lines:
            stripped_line = line.strip()
            
            # Skip empty lines and comment-only lines
            if not stripped_line or stripped_line.startswith('--'):
                continue
            
            # Handle dollar quoting
            if '$$' in line:
                if not in_dollar_quote:
                    # Starting dollar quote
                    in_dollar_quote = True
                    # Extract the tag if any (e.g., $tag$)
                    parts = line.split('$$')
                    if len(parts) >= 2:
                        dollar_tag = parts[0].split('$')[-1] if '$' in parts[0] else ''
                elif in_dollar_quote:
                    # Check if this ends the dollar quote
                    if f'$${dollar_tag}$$' in line or (dollar_tag == '' and '$$' in line):
                        in_dollar_quote = False
                        dollar_tag = None
            
            current_statement.append(line)
            
            # If we're not in a dollar quote and the line ends with semicolon, end statement
            if not in_dollar_quote and stripped_line.endswith(';'):
                statement_text = '\n'.join(current_statement).strip()
                if statement_text:
                    # Remove comments from the statement
                    clean_statement = self._clean_statement(statement_text)
                    if clean_statement:
                        statements.append(clean_statement)
                current_statement = []
        
        # Handle any remaining statement
        if current_statement:
            statement_text = '\n'.join(current_statement).strip()
            if statement_text:
                clean_statement = self._clean_statement(statement_text)
                if clean_statement:
                    statements.append(clean_statement)
        
        return statements
    
    def _clean_statement(self, statement: str) -> str:
        """Remove comments and clean up a SQL statement"""
        lines = statement.split('\n')
        sql_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('--'):
                sql_lines.append(line)
        
        return '\n'.join(sql_lines).strip()

    def save_schema_file(self, schema_ddl: str, output_path: str) -> bool:
        """Save the generated schema to a file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(schema_ddl)
            logger.info(f"Schema saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving schema file: {e}")
            return False
    
    def convert_schema(self, source_db_path: str, target_db_config: Dict[str, Any]) -> str:
        """
        Main method to convert schema from source to target database
        Returns the path to the generated schema file
        """
        try:
            logger.info("Starting schema conversion process...")
            
            # Analyze source schema
            schema_info = self.analyze_source_schema()
            
            # Generate PostgreSQL schema DDL
            schema_ddl = self.generate_postgresql_schema(schema_info)
            
            # Save schema to file
            output_path = "scripts/migration/schema/postgresql_schema.sql"
            if not self.save_schema_file(schema_ddl, output_path):
                raise Exception("Failed to save schema file")
            
            # Execute schema on target database
            logger.info("Executing schema on target database...")
            with self.target_engine.connect() as conn:
                with conn.begin():  # Use transaction
                    # Parse DDL into individual statements, handling $$ delimited functions
                    statements = self._parse_ddl_statements(schema_ddl)
                    for statement in statements:
                        if statement:
                            try:
                                from sqlalchemy import text
                                conn.execute(text(statement))
                                logger.info(f"Executed: {statement[:50]}...")
                            except Exception as e:
                                logger.error(f"Statement execution failed: {e}")
                                logger.error(f"Failed statement: {statement}")
                                raise  # Re-raise to trigger rollback
            
            logger.info("Schema conversion completed successfully")
            return output_path
            
        except Exception as e:
            logger.error(f"Schema conversion failed: {e}")
            raise

def main():
    """Main conversion process"""
    logger.info("Starting PostgreSQL schema conversion...")
    
    try:
        # Configuration
        source_db_uri = DatabaseConfig.SQLALCHEMY_DATABASE_URI
        target_db_uri = os.getenv('POSTGRES_URL', 'postgresql://hrp_user:password@localhost:5432/hrp_database')
        
        # Initialize converter
        converter = PostgreSQLSchemaConverter(source_db_uri, target_db_uri)
        
        # Analyze source schema
        schema_info = converter.analyze_source_schema()
        
        # Generate PostgreSQL schema
        postgresql_schema = converter.generate_postgresql_schema(schema_info)
        
        # Validate conversion
        validation_results = converter.validate_conversion(schema_info)
        
        # Save schema file
        output_path = "scripts/migration/schema/generated_postgresql_schema.sql"
        converter.save_schema_file(postgresql_schema, output_path)
        
        # Generate conversion report
        report = {
            'conversion_timestamp': datetime.now().isoformat(),
            'source_database': source_db_uri,
            'target_database': target_db_uri,
            'validation_results': validation_results,
            'output_file': output_path
        }
        
        # Save report
        import json
        report_path = "scripts/migration/schema/conversion_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Schema conversion completed successfully!")
        logger.info(f"Generated schema: {output_path}")
        logger.info(f"Conversion report: {report_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Schema conversion failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)