from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError, APITimeoutError
from typing import Dict, Optional, List
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Generation API",
    description="API for generating text using OpenAI",
    version="1.0.0"
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DetailParams(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)

class Action(BaseModel):
    params: DetailParams
    detailParams: Optional[Dict] = None

class RequestBody(BaseModel):
    action: Action

class TextOutput(BaseModel):
    text: str

class SimpleTextOutput(BaseModel):
    simpleText: TextOutput

class ResponseTemplate(BaseModel):
    version: str = "2.0"
    template: Dict[str, List[SimpleTextOutput]]

@app.post(
    "/generate",
    response_model=ResponseTemplate,
    summary="Generate text using OpenAI",
    responses={
        200: {"description": "Successful text generation"},
        400: {"description": "Invalid input parameters"},
        429: {"description": "API rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
async def generate_text(request: RequestBody):
    try:
        prompt = request.action.params.prompt
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )

        generated_text: str = response.choices[0].message.content or ""

        if not generated_text:
            raise HTTPException(status_code=500, detail="Generated text is empty")

        return ResponseTemplate(
            version="2.0",
            template={
                "outputs": [
                    SimpleTextOutput(simpleText=TextOutput(text=generated_text))
                ]
            }
        )
    except APITimeoutError as e:
        logger.error(f"OpenAI API timeout: {str(e)}")
        raise HTTPException(status_code=429, detail="OpenAI API timeout occurred")
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail="OpenAI API error occurred")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
