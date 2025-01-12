-- Drop existing table and indexes if they exist
DROP TABLE IF EXISTS index_logs;
DROP INDEX IF EXISTS idx_source_checksum;

-- Create index logs table
CREATE TABLE IF NOT EXISTS index_logs (
    id VARCHAR(255) PRIMARY KEY,
    source VARCHAR(1024) NOT NULL,
    source_type VARCHAR(128) NOT NULL,
    checksum VARCHAR(255) NOT NULL,
    status VARCHAR(128) NOT NULL,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    created_at TIMESTAMP NOT NULL,
    created_by VARCHAR(128) NOT NULL,
    modified_at TIMESTAMP NOT NULL,
    modified_by VARCHAR(128) NOT NULL,
    CONSTRAINT uix_source_source_type UNIQUE (source, source_type),
    CONSTRAINT uix_checksum UNIQUE (checksum)
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_source_checksum ON index_logs (source, source_type, checksum);
CREATE INDEX IF NOT EXISTS idx_status ON index_logs (status);
CREATE INDEX IF NOT EXISTS idx_created_by ON index_logs (created_by);
CREATE INDEX IF NOT EXISTS idx_modified_by ON index_logs (modified_by);

-- Add comment to table
COMMENT ON TABLE index_logs IS 'Table for tracking document indexing status and metadata';

-- Add comments to columns
COMMENT ON COLUMN index_logs.id IS 'Unique identifier for the index log record';
COMMENT ON COLUMN index_logs.source IS 'Source path or URL of the document';
COMMENT ON COLUMN index_logs.source_type IS 'Type of the document (pdf, csv, json, etc.)';
COMMENT ON COLUMN index_logs.checksum IS 'SHA-256 hash of the document content';
COMMENT ON COLUMN index_logs.status IS 'Current status of the indexing process (PENDING, IN_PROGRESS, COMPLETED, FAILED)';
COMMENT ON COLUMN index_logs.error_message IS 'Error message if indexing failed';
COMMENT ON COLUMN index_logs.created_at IS 'Timestamp when the record was created';
COMMENT ON COLUMN index_logs.created_by IS 'User ID who created the record';
COMMENT ON COLUMN index_logs.modified_at IS 'Timestamp when the record was last modified';
COMMENT ON COLUMN index_logs.modified_by IS 'User ID who last modified the record';