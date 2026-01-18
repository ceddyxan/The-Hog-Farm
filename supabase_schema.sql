-- Supabase Database Schema for Hog Farm Application
-- Run this SQL in your Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Hogs table - stores hog information
CREATE TABLE IF NOT EXISTS hogs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    hog_id INTEGER UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Weight measurements table
CREATE TABLE IF NOT EXISTS weight_measurements (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    hog_id INTEGER NOT NULL REFERENCES hogs(hog_id) ON DELETE CASCADE,
    measurement_date DATE NOT NULL,
    weight_kg DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(hog_id, measurement_date)
);

-- Financial transactions table
CREATE TABLE IF NOT EXISTS financial_transactions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    transaction_date DATE NOT NULL,
    transaction_type VARCHAR(50) NOT NULL, -- 'sale', 'purchase', 'expense'
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    description TEXT,
    hog_id INTEGER REFERENCES hogs(hog_id) ON DELETE SET NULL,
    weight_kg DECIMAL(10,2),
    price_per_kg DECIMAL(10,2),
    amount DECIMAL(12,2) NOT NULL,
    buyer VARCHAR(200),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Budgets table
CREATE TABLE IF NOT EXISTS budgets (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    budget_name VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    planned_amount DECIMAL(12,2) NOT NULL,
    actual_amount DECIMAL(12,2) DEFAULT 0,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit trail table
CREATE TABLE IF NOT EXISTS audit_trail (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    action VARCHAR(100) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    record_id UUID,
    old_values JSONB,
    new_values JSONB,
    user_id VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_weight_measurements_hog_id ON weight_measurements(hog_id);
CREATE INDEX IF NOT EXISTS idx_weight_measurements_date ON weight_measurements(measurement_date);
CREATE INDEX IF NOT EXISTS idx_financial_transactions_date ON financial_transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_financial_transactions_type ON financial_transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_audit_trail_timestamp ON audit_trail(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_trail_table ON audit_trail(table_name);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update updated_at
CREATE TRIGGER update_hogs_updated_at BEFORE UPDATE ON hogs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_weight_measurements_updated_at BEFORE UPDATE ON weight_measurements
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_financial_transactions_updated_at BEFORE UPDATE ON financial_transactions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_budgets_updated_at BEFORE UPDATE ON budgets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE hogs ENABLE ROW LEVEL SECURITY;
ALTER TABLE weight_measurements ENABLE ROW LEVEL SECURITY;
ALTER TABLE financial_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE budgets ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_trail ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users (you can restrict this later)
CREATE POLICY "Enable all operations for authenticated users" ON hogs
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Enable all operations for authenticated users" ON weight_measurements
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Enable all operations for authenticated users" ON financial_transactions
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Enable all operations for authenticated users" ON budgets
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Enable all operations for authenticated users" ON audit_trail
    FOR ALL USING (auth.role() = 'authenticated');

-- Insert sample data (optional - you can remove this)
-- This will help you test the migration
INSERT INTO hogs (hog_id) VALUES 
(1), (2), (3), (4)
ON CONFLICT (hog_id) DO NOTHING;
