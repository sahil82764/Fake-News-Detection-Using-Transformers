from fastapi import APIRouter, HTTPException
from ..schemas import NewsArticle, PredictionResponse
from ..models.predictor import predictor # Import the singleton predictor instance

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict_news(article: NewsArticle):
    """
    Endpoint to predict if a news article is real or fake.
    
    - **text**: The text of the news article to analyze.
    """
    if not article.text or not article.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        label, confidence = predictor.predict(article.text)
        return PredictionResponse(prediction=label, confidence=confidence)
    except Exception as e:
        # Log the error here in a real application
        print(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
