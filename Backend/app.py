from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Replace with the module where your model is defined
from Backend.model import load_models, answer

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

models = load_models()

class Query(BaseModel):
    question: str
    # You can use this to pass site context if needed
    # site_context: str

@app.post("/ask/")
async def ask_question(query: Query):
    try:
        ans = answer(query.question, models['tools'])
        return {"answer": ans}
    except Exception as e:
        logging.error(f"Exception occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get('/')
def running():
    return {"status": "working"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

