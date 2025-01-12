-- Drop existing table and indexes if they exist
DROP TABLE IF EXISTS distributed_locks;
DROP INDEX IF EXISTS idx_lock_key;

-- Create distributed locks table
CREATE TABLE IF NOT EXISTS distributed_locks (
    id VARCHAR(255) PRIMARY KEY,
    lock_key VARCHAR(255) NOT NULL,
    instance_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    CONSTRAINT uix_lock_key UNIQUE (lock_key)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_lock_key ON distributed_locks (lock_key, instance_name);

-- Add comment to table
COMMENT ON TABLE distributed_locks IS 'Table for managing distributed locks across application instances';

-- Add comments to columns
COMMENT ON COLUMN distributed_locks.id IS 'Unique identifier for the lock record';
COMMENT ON COLUMN distributed_locks.lock_key IS 'Unique key representing the resource being locked';
COMMENT ON COLUMN distributed_locks.instance_name IS 'Name/ID of the application instance holding the lock';
COMMENT ON COLUMN distributed_locks.created_at IS 'Timestamp when the lock was acquired'; 