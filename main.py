from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, OpenAIError, APITimeoutError
from typing import Dict, Optional, List
import os
from contextlib import asynccontextmanager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 앱 및 OpenAI 클라이언트 설정 ---
# lifespan 관리자를 사용하여 시작 시 클라이언트를 초기화합니다.
# 이는 최신 FastAPI에서 권장하는 방식입니다.
openai_client: Optional[AsyncOpenAI] = None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱의 생명주기 동안 OpenAI 클라이언트를 관리합니다.
    앱 시작 시 클라이언트를 초기화하고, 종료 시 리소스를 정리합니다.
    """
    global openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    
    openai_client = AsyncOpenAI(api_key=api_key)
    logger.info(f"OpenAI client initialized with model: {OPENAI_MODEL}")
    
    yield
    
    # 앱 종료 시 정리할 내용이 있다면 여기에 추가합니다.
    logger.info("Shutting down OpenAI client.")


app = FastAPI(
    title="Text Generation API",
    description="API for generating text using OpenAI",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic 모델 정의 ---
# 참고: 아래 모델 구조(RequestBody, Action 등)는 특정 플랫폼(예: 챗봇 서비스)과의
# 연동을 위해 설계된 것으로 보입니다. 'detailParams' 필드는 현재 로직에서
# 사용되지 않지만, 호환성을 위해 유지되었습니다.

class DetailParams(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="텍스트 생성을 위한 입력 프롬프트")

class Action(BaseModel):
    params: DetailParams
    detailParams: Optional[Dict] = None # 현재 로직에서는 사용되지 않는 필드

class RequestBody(BaseModel):
    action: Action

class TextOutput(BaseModel):
    text: str

class SimpleTextOutput(BaseModel):
    simpleText: TextOutput

class ResponseTemplate(BaseModel):
    version: str = "2.0"
    template: Dict[str, List[SimpleTextOutput]]

# --- API 엔드포인트 ---
@app.post(
    "/generate",
    response_model=ResponseTemplate,
    summary="OpenAI를 사용하여 텍스트 생성",
    responses={
        200: {"description": "성공적인 텍스트 생성"},
        400: {"description": "잘못된 입력 매개변수"},
        429: {"description": "API 속도 제한 초과"},
        500: {"description": "내부 서버 오류"}
    }
)
async def generate_text(request: RequestBody):
    """
    사용자의 프롬프트를 기반으로 OpenAI API를 사용하여 텍스트를 생성합니다.
    """
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI client is not available")

    try:
        prompt = request.action.params.prompt
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )

        generated_text = response.choices[0].message.content or ""

        if not generated_text.strip():
            raise HTTPException(status_code=500, detail="Generated text is empty or contains only whitespace.")

        return ResponseTemplate(
            template={
                "outputs": [
                    SimpleTextOutput(simpleText=TextOutput(text=generated_text))
                ]
            }
        )
    except APITimeoutError as e:
        logger.error(f"OpenAI API timeout: {e}")
        raise HTTPException(status_code=429, detail="OpenAI API request timed out")
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred with the OpenAI API: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred")

@app.get("/health", summary="Health Check", tags=["Monitoring"])
async def health_check():
    """
    API가 정상적으로 실행 중인지 확인하는 간단한 상태 확인 엔드포인트입니다.
    """
    return {"status": "ok"}
