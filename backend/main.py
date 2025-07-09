from fastapi import FastAPI
from .api import predict # Import the router from api/predict.py

# Create the FastAPI app instance
app = FastAPI(
    title="Fake News Detection API",
    description="An API to classify news articles as Real or Fake using a DistilBERT model.",
    version="1.0.0"
)

# Include the prediction router
# All routes defined in api/predict.py will be included under the /api prefix
app.include_router(predict.router, prefix="/api")

@app.get("/", tags=["Root"])
def read_root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Fake News Detection API. Go to /docs for documentation."}
