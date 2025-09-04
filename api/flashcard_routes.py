import logging

from fastapi import APIRouter, HTTPException

from core.models import FlashcardRequest
from services.flashcard_service import FlashcardsService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/flashcards", tags=["Flashcards"])
service = FlashcardsService()


# ✅ Generate flashcards
@router.post("/generate")
async def generate_flashcards(request: FlashcardRequest):
    try:
        logging.info(f"Incoming request: {request}")
        response = service.generate_and_save_flashcards(request)
        logging.info(f"Generated flashcards: {response}")
        return response
    except Exception as e:
        logging.error(f"Error in generate_flashcards: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Review flashcard (changed POST → PUT)
@router.put("/{card_id}/review")
async def review_flashcard(card_id: str, correct: bool, difficulty_rating: int = None):
    try:
        response = service.record_review(card_id, correct, difficulty_rating)
        return {"status": "success", "card_id": card_id, "review": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Get flashcard statistics
@router.get("/stats")
async def get_flashcard_stats():
    try:
        stats = service.get_flashcard_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
