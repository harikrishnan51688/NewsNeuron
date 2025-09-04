import json
import uuid
from datetime import datetime

from psycopg2.extras import execute_values

from core.models import FlashcardRequest, FlashcardStatus, SourceType
from retrieval.vector_store import VectorStore
from services.spaced_repetition import calculate_next_review


class FlashcardsService:
    def __init__(self):
        self.store = VectorStore()  # Database connection

    def save_flashcards(self, flashcards, request: FlashcardRequest):
        """Save generated flashcards into the database."""
        values = []
        for card in flashcards:
            values.append((
                card.get("id", str(uuid.uuid4())),
                card.get("category", request.topic or "General"),
                card["question"],
                card["answer"],
                request.difficulty.value,
                json.dumps(card.get("tags", [])),
                FlashcardStatus.NEW.value,
                datetime.now(),
                0.0,
                request.source_type.value,
                datetime.now() if request.source_type == SourceType.RECENT_NEWS else None,
                card.get("readMore"),
                card.get("context")
            ))

        with self.store.pg_conn.cursor() as cursor:
            execute_values(cursor, """
                INSERT INTO flashcards (
                    id, category, question, answer, difficulty, tags,
                    status, created_at, success_rate, source_type, news_date,
                    source_url, context
                ) VALUES %s
                ON CONFLICT (id) DO UPDATE
                SET question = EXCLUDED.question,
                    answer = EXCLUDED.answer,
                    tags = EXCLUDED.tags,
                    context = EXCLUDED.context
            """, values)
            self.store.pg_conn.commit()

    def _generate_flashcards(self, request: FlashcardRequest):
        """Generate dummy flashcards for now — replace with AI logic later."""
        return [
            {
                "id": str(uuid.uuid4()),
                "category": request.topic or "General",
                "question": f"What is {request.topic}?",
                "answer": f"{request.topic} is an important concept.",
                "tags": [request.topic],
                "readMore": None,
                "context": None
            },
            {
                "id": str(uuid.uuid4()),
                "category": request.topic or "General",
                "question": f"Explain key points about {request.topic}.",
                "answer": f"Some key points about {request.topic} include ...",
                "tags": [request.topic],
                "readMore": None,
                "context": None
            }
        ]

    def generate_and_save_flashcards(self, request: FlashcardRequest):
        """Generate flashcards and save them in the database."""
        flashcards = self._generate_flashcards(request)
        self.save_flashcards(flashcards, request)
        return {
            "message": "Flashcards generated & saved successfully",
            "count": len(flashcards),
            "flashcards": flashcards
        }

    def record_review(self, card_id: str, correct: bool, difficulty_rating: int = None):
        """Update review stats for a specific flashcard."""
        with self.store.pg_conn.cursor() as cursor:
            cursor.execute("""
                SELECT review_count, success_rate, status, difficulty
                FROM flashcards WHERE id = %s
            """, (card_id,))
            row = cursor.fetchone()

            if not row:
                return {"success": False, "error": "Flashcard not found"}

            review_count, success_rate, status, difficulty = row
            review_count = review_count or 0
            success_rate = float(success_rate or 0.0)
            new_success_rate = (success_rate * review_count + (1.0 if correct else 0.0)) / (review_count + 1)

            next_review, new_status = calculate_next_review(
                correct, review_count, new_success_rate, difficulty, difficulty_rating
            )

            cursor.execute("""
                UPDATE flashcards
                SET last_reviewed = %s,
                    review_count = review_count + 1,
                    success_rate = %s,
                    status = %s,
                    next_review = %s
                WHERE id = %s
            """, (datetime.now(), new_success_rate, new_status.value, next_review, card_id))

            self.store.pg_conn.commit()

        return {
            "success": True,
            "new_status": new_status.value,
            "next_review": next_review,
            "success_rate": new_success_rate
        }

    def get_flashcard_stats(self):
        """Fetch flashcard statistics from the database."""
        with self.store.pg_conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) AS total,
                    SUM(CASE WHEN review_count > 0 THEN 1 ELSE 0 END) AS reviewed,
                    SUM(CASE WHEN success_rate = 1 THEN 1 ELSE 0 END) AS correct,
                    SUM(CASE WHEN success_rate < 1 AND review_count > 0 THEN 1 ELSE 0 END) AS wrong
                FROM flashcards
            """)
            result = cursor.fetchone()

        total, reviewed, correct, wrong = result
        return {
            "total": total or 0,
            "reviewed": reviewed or 0,
            "correct": correct or 0,
            "wrong": wrong or 0,
            "remaining": (total or 0) - (reviewed or 0)
        }
