-- PostgreSQL Schema Generated from SQLAlchemy Models
-- Generated on: 2025-09-18T10:19:58.159216

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types
CREATE TYPE race_status AS ENUM ('upcoming', 'running', 'completed', 'cancelled');
CREATE TYPE horse_sex AS ENUM ('M', 'F', 'G', 'C');
CREATE TYPE surface_type AS ENUM ('dirt', 'turf', 'synthetic', 'all_weather');
CREATE TYPE track_condition AS ENUM ('fast', 'good', 'firm', 'soft', 'heavy', 'sloppy', 'muddy', 'frozen');

-- Table: races
CREATE TABLE races (
    id SERIAL PRIMARY KEY,
    name TEXT,
    date TEXT,
    time TEXT,
    track TEXT,
    distance DECIMAL(12,2),
    surface TEXT,
    race_class TEXT,
    prize_money DECIMAL(12,2),
    weather TEXT,
    track_condition TEXT,
    status TEXT,
    results TEXT,
    created_at TEXT,
    updated_at TEXT
);

-- Indexes for performance optimization
CREATE INDEX idx_races_date ON races(date);
CREATE INDEX idx_races_track ON races(track);
CREATE INDEX idx_races_status ON races(status);
CREATE INDEX idx_races_date_status ON races(date, status);

-- Additional constraints
ALTER TABLE races ADD CONSTRAINT chk_distance CHECK (distance > 0);
ALTER TABLE races ADD CONSTRAINT chk_prize_money CHECK (prize_money >= 0);

-- Automatic timestamp update function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_races_updated_at BEFORE UPDATE ON races
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
