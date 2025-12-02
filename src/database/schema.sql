-- Create database
CREATE DATABASE bank_reviews;

-- Connect to database
\c bank_reviews;

-- Create Banks table
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL,
    app_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Reviews table
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INTEGER REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text TEXT,
    rating FLOAT,
    review_date DATE,
    sentiment_label VARCHAR(50),
    sentiment_score FLOAT,
    source VARCHAR(50),
    cleaned_text TEXT,
    keywords TEXT[],
    theme VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_score);
CREATE INDEX idx_reviews_date ON reviews(review_date);
CREATE INDEX idx_banks_name ON banks(bank_name);

-- Create view for analysis
CREATE VIEW bank_sentiment_summary AS
SELECT 
    b.bank_name,
    COUNT(r.review_id) as total_reviews,
    AVG(r.rating) as avg_rating,
    AVG(r.sentiment_score) as avg_sentiment,
    SUM(CASE WHEN r.sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
    SUM(CASE WHEN r.sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
    SUM(CASE WHEN r.sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
FROM banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name;