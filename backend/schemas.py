from pydantic import BaseModel

class NewsArticle(BaseModel):
    """
    Pydantic model for the input data.
    Ensures that any request to the /predict endpoint must contain a 'text' field.
    """
    text: str

class PredictionResponse(BaseModel):
    """
    Pydantic model for the output data.
    Defines the structure of the JSON response.
    """
    prediction: str  # e.g., "Real" or "Fake"
    confidence: float # e.g., 0.98
