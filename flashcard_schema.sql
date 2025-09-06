-- Drop existing tables if needed
DROP TABLE IF EXISTS study_results CASCADE;
DROP TABLE IF EXISTS study_sessions CASCADE;
DROP TABLE IF EXISTS flashcards CASCADE;
DROP TABLE IF EXISTS flashcard_sets CASCADE;
DROP TABLE IF EXISTS source_articles CASCADE;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Source articles table
CREATE TABLE source_articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    url TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Flashcard sets
CREATE TABLE flashcard_sets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Flashcards table (matches API expectations)
CREATE TABLE flashcards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    explanation TEXT,
    category TEXT,
    difficulty TEXT NOT NULL DEFAULT 'medium', -- 'easy', 'medium', 'hard'
    flashcard_type TEXT NOT NULL DEFAULT 'basic', -- 'basic', 'multiple_choice', etc.
    source_article_id UUID REFERENCES source_articles(id) ON DELETE SET NULL,
    source_url TEXT,
    entities TEXT[] DEFAULT '{}', -- Array of entities
    tags TEXT[] DEFAULT '{}', -- Array of tags
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    correct_count INTEGER DEFAULT 0,
    incorrect_count INTEGER DEFAULT 0,
    last_studied TIMESTAMP,
    mastery_level REAL DEFAULT 0.0 -- Float between 0.0 and 1.0
);

-- Study sessions (matches API expectations)
CREATE TABLE study_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    started_at TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP,
    total_cards INTEGER DEFAULT 0,
    cards_correct INTEGER DEFAULT 0,
    cards_incorrect INTEGER DEFAULT 0,
    category_filter TEXT,
    difficulty_filter TEXT
);

-- Study results (matches API expectations)
CREATE TABLE study_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES study_sessions(id) ON DELETE CASCADE,
    flashcard_id UUID REFERENCES flashcards(id) ON DELETE CASCADE,
    is_correct BOOLEAN NOT NULL,
    confidence_level INTEGER, -- 1-5 scale
    time_spent_seconds INTEGER,
    attempted_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for better performance
CREATE INDEX idx_flashcards_category ON flashcards(category);
CREATE INDEX idx_flashcards_difficulty ON flashcards(difficulty);
CREATE INDEX idx_flashcards_flashcard_type ON flashcards(flashcard_type);
CREATE INDEX idx_flashcards_mastery_level ON flashcards(mastery_level);
CREATE INDEX idx_flashcards_last_studied ON flashcards(last_studied);
CREATE INDEX idx_flashcards_entities ON flashcards USING GIN(entities);
CREATE INDEX idx_flashcards_tags ON flashcards USING GIN(tags);

-- Trigger to update updated_at on flashcards
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_flashcards_updated_at 
    BEFORE UPDATE ON flashcards 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_flashcard_sets_updated_at 
    BEFORE UPDATE ON flashcard_sets 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_source_articles_updated_at 
    BEFORE UPDATE ON source_articles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert some seed data
INSERT INTO source_articles (title, url, content) VALUES 
('Sample Article 1', 'https://example.com/article1', 'This is sample content for article 1.'),
('Sample Article 2', 'https://example.com/article2', 'This is sample content for article 2.');

INSERT INTO flashcard_sets (name, description) VALUES 
('General Knowledge', 'Basic trivia flashcards'),
('Science Facts', 'Scientific concepts and facts');

-- Get the IDs for seed data
INSERT INTO flashcards (question, answer, explanation, category, difficulty, flashcard_type, entities, tags)
VALUES 
('What is the capital of France?', 'Paris', 'Paris has been the capital of France since 508 AD.', 'Geography', 'easy', 'basic', ARRAY['France', 'Paris'], ARRAY['geography', 'capitals']),
('Who wrote "Hamlet"?', 'William Shakespeare', 'Shakespeare wrote Hamlet around 1600-1601.', 'Literature', 'medium', 'basic', ARRAY['Shakespeare', 'Hamlet'], ARRAY['literature', 'plays']),
('What is the speed of light?', '299,792,458 meters per second', 'The speed of light in a vacuum is a fundamental physical constant.', 'Physics', 'hard', 'basic', ARRAY['light', 'physics'], ARRAY['physics', 'constants']),
('What is H2O?', 'Water', 'H2O is the chemical formula for water.', 'Chemistry', 'easy', 'basic', ARRAY['water', 'H2O'], ARRAY['chemistry', 'molecules']);

-- Views for statistics
CREATE OR REPLACE VIEW flashcard_stats AS
SELECT 
    category,
    COUNT(*) AS total_flashcards,
    AVG(mastery_level) AS avg_mastery,
    AVG(CASE 
        WHEN (correct_count + incorrect_count) > 0 
        THEN correct_count::REAL / (correct_count + incorrect_count) * 100
        ELSE 0 
    END) AS avg_accuracy_rate
FROM flashcards
WHERE category IS NOT NULL
GROUP BY category;

CREATE OR REPLACE VIEW difficulty_stats AS
SELECT 
    difficulty,
    COUNT(*) AS total_flashcards,
    AVG(mastery_level) AS avg_mastery,
    COUNT(CASE WHEN mastery_level >= 0.8 THEN 1 END) AS mastered_count
FROM flashcards
GROUP BY difficulty;

-- Function to calculate next review date based on mastery level
CREATE OR REPLACE FUNCTION next_review_date(mastery_level REAL, last_studied TIMESTAMP)
RETURNS TIMESTAMP AS $$
BEGIN
    IF last_studied IS NULL THEN
        RETURN NOW();
    END IF;
    
    -- Spaced repetition intervals based on mastery
    CASE 
        WHEN mastery_level < 0.3 THEN RETURN last_studied + INTERVAL '1 day';
        WHEN mastery_level < 0.5 THEN RETURN last_studied + INTERVAL '3 days';
        WHEN mastery_level < 0.7 THEN RETURN last_studied + INTERVAL '1 week';
        WHEN mastery_level < 0.9 THEN RETURN last_studied + INTERVAL '2 weeks';
        ELSE RETURN last_studied + INTERVAL '1 month';
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- View for cards due for review
CREATE OR REPLACE VIEW cards_due_for_review AS
SELECT 
    *,
    next_review_date(mastery_level, last_studied) as next_review
FROM flashcards
WHERE next_review_date(mastery_level, last_studied) <= NOW()
ORDER BY mastery_level ASC, last_studied ASC NULLS FIRST;