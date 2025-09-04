from datetime import datetime, timedelta
from core.models import FlashcardStatus

def calculate_next_review(correct: bool, review_count: int, success_rate: float,
                          difficulty: str = "medium", difficulty_rating: int = None):
    """
    Enhanced spaced repetition scheduling.
    """
    difficulty_modifiers = {"easy": 1.2, "medium": 1.0, "hard": 0.8}
    card_modifier = difficulty_modifiers.get(difficulty, 1.0)

    if correct:
        if review_count == 0:
            interval_days = 1
        elif review_count == 1:
            interval_days = 3
        elif review_count == 2:
            interval_days = 7
        else:
            base_interval = min(90, 14 * (1.4 ** (review_count - 3)))
            rating_modifier = {1: 1.8, 2: 1.3, 3: 1.0, 4: 0.7, 5: 0.5}.get(difficulty_rating, 1.0)
            interval_days = max(1, int(base_interval * success_rate * card_modifier * rating_modifier))
    else:
        interval_days = 1 if success_rate < 0.3 else 2 if success_rate < 0.6 else 3

    next_review = datetime.now() + timedelta(days=interval_days)

    if success_rate >= 0.9 and review_count >= 3:
        status = FlashcardStatus.KNOWN
    elif success_rate >= 0.7:
        status = FlashcardStatus.LEARNING
    else:
        status = FlashcardStatus.REVIEW

    return next_review, status
