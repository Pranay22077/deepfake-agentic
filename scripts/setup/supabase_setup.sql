-- Create video_analyses table for tracking deepfake analysis results
CREATE TABLE IF NOT EXISTS video_analyses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    filename TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    prediction TEXT NOT NULL CHECK (prediction IN ('real', 'fake')),
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    models_used TEXT[] NOT NULL DEFAULT '{}',
    processing_time DECIMAL(6,2) NOT NULL DEFAULT 0,
    analysis_result JSONB,
    user_ip TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_video_analyses_created_at ON video_analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_video_analyses_prediction ON video_analyses(prediction);
CREATE INDEX IF NOT EXISTS idx_video_analyses_confidence ON video_analyses(confidence);

-- Enable Row Level Security (RLS)
ALTER TABLE video_analyses ENABLE ROW LEVEL SECURITY;

-- Create policy to allow public read access (for analytics)
CREATE POLICY "Allow public read access" ON video_analyses
    FOR SELECT USING (true);

-- Create policy to allow public insert (for saving analyses)
CREATE POLICY "Allow public insert" ON video_analyses
    FOR INSERT WITH CHECK (true);

-- Grant permissions to anon role
GRANT SELECT, INSERT ON video_analyses TO anon;
GRANT USAGE ON SCHEMA public TO anon;