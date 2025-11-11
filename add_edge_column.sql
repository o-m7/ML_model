-- Add edge column to trades table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='trades' AND column_name='edge') THEN
        ALTER TABLE trades ADD COLUMN edge DECIMAL(5,4);
        RAISE NOTICE 'Added edge column to trades table';
    ELSE
        RAISE NOTICE 'Edge column already exists in trades table';
    END IF;
END $$;

