import os
import time
import base64
import binascii
import logging
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# =====================================================
# NEST AI ASSISTANT
# Flutter + FastAPI + Gemini
# =====================================================

load_dotenv()

app = FastAPI(title="NEST AI Assistant")

# Allow Flutter Web running locally on Edge/Chrome with changing dev ports.
# For deployment, replace this regex with your actual HTTPS frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nest-75993.web.app",
        "https://nest-75993.firebaseapp.com",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("nest-ai")

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_HISTORY_MESSAGES = 20

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

SUPPORTED_LANGUAGES = {
    "English": "English",
    "Korean": "Korean",
    "Mongolian": "Mongolian",
    "Myanmar": "Burmese (Myanmar)",
    "Burmese": "Burmese (Myanmar)",
    "en": "English",
    "ko": "Korean",
    "mn": "Mongolian",
    "my": "Burmese (Myanmar)",
}

ALLOWED_VISAS = {
    "D-2",
    "D-4",
    "F-2",
    "F-4",
}


# =====================================================
# REQUEST MODELS
# =====================================================

class Message(BaseModel):
    role: Literal["user", "model"]
    text: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=4000)
    language: str = "English"
    visa_type: str = "D-2"

    history: list[Message] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_MESSAGES,
    )

    image_base64: str | None = None
    image_mime_type: str | None = None


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
def home():
    return {
        "status": "NEST AI backend is running",
        "service": "NEST AI Chat Assistant",
        "photo_support": True,
    }


# =====================================================
# IMAGE VALIDATION
# =====================================================

def validate_image(request: ChatRequest):

    if not request.image_base64:
        return None

    mime_type = request.image_mime_type

    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Only JPEG, PNG, and WebP images "
                "are supported."
            ),
        )

    max_base64_length = (
        ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 16
    )

    if len(request.image_base64) > max_base64_length:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 8 MB.",
        )

    try:
        image_bytes = base64.b64decode(
            request.image_base64,
            validate=True,
        )

    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Invalid image data.",
        )

    if not image_bytes:
        raise HTTPException(
            status_code=422,
            detail="The image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 8 MB.",
        )

    is_jpeg = image_bytes.startswith(
        b"\xff\xd8\xff"
    )

    is_png = image_bytes.startswith(
        b"\x89PNG\r\n\x1a\n"
    )

    is_webp = (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    )

    valid_signature = (
        (mime_type == "image/jpeg" and is_jpeg)
        or (mime_type == "image/png" and is_png)
        or (mime_type == "image/webp" and is_webp)
    )

    if not valid_signature:
        raise HTTPException(
            status_code=422,
            detail=(
                "Image format does not match "
                "its content."
            ),
        )

    return image_bytes


# =====================================================
# SYSTEM INSTRUCTION
# =====================================================

def build_system_instruction(
    language: str,
    visa_type: str,
) -> str:

    return f"""
You are NEST AI, a friendly and reliable settlement
assistant for foreigners living in South Korea.

USER PROFILE:
- Preferred language: {language}
- Selected visa type: {visa_type}

YOUR PURPOSE:
Help foreigners understand:
- Korean housing and rental contracts
- Moving and address registration
- Immigration and administrative procedures
- Visa-related information
- University life
- Healthcare and insurance
- Employment and part-time work
- Everyday life in Korea

LANGUAGE RULES (IMPORTANT):
1. NEST supports exactly four conversation languages:
   English, Korean, Mongolian, and Burmese (Myanmar).
2. For each new user message, identify which of those
   four languages the user is using. Reply in that language,
   even if it differs from the app language ({language}).
3. If the user explicitly asks for a reply in one of these
   four languages, follow that request instead.
4. If the message is ambiguous (for example, only a number,
   a greeting shared by languages, or an image with no clear
   language request), use the app language ({language}).
5. A document may contain Korean text while the user asks
   about it in Mongolian, English, or Burmese. Reply in the
   user's requested language; do not switch to the document's
   language unless requested. Keep Korean terms where helpful.
6. If the user writes in an unsupported language, politely
   ask them to use English, Korean, Mongolian, or Burmese.
   Do not continue a full conversation in unsupported languages.
7. Never refuse to speak any of the four supported languages
   based on app settings, previous replies, or conversation history.
   A new user message can change the response language.
8. For translation requests, translate faithfully between the
   four supported languages and preserve the original meaning.
9. Use simple, beginner-friendly language and explain unfamiliar
   Korean terms. Do not invent language restrictions.

RESPONSE LENGTH AND STYLE:
1. Keep ordinary answers concise, usually
   around 100-150 words.
2. Answer the user's question directly first.
3. Avoid unnecessary introductions.
4. Do not repeat the user's question.
5. Use short paragraphs and bullet points
   when they improve readability.
6. Give numbered steps for procedures.
7. Avoid repeating the same advice.
8. For complicated procedures, explain the
   essential information before extra details.
9. For document translation or analysis,
   provide enough detail to explain important
   information accurately.
10. If the user explicitly requests a detailed
    explanation, respond in greater detail.
11. If a question can be answered in one or
    two sentences, keep it that short.
12. End naturally without unnecessary
    follow-up questions.

VISA AND ADMINISTRATIVE GUIDANCE:
1. Consider the selected visa type when relevant.
2. Do not assume the visa type alone determines
   all legal requirements.
3. Never invent laws, deadlines, fees, or
   administrative requirements.
4. Explain when requirements depend on
   individual circumstances.
5. Clearly distinguish confirmed general
   information from uncertain details.
6. Recommend official verification for
   requirements that may change.
7. Never claim to have checked a live
   official source.
8. Do not request sensitive identification
   details, bank information, or passwords.

OFFICIAL INFORMATION:
For immigration-related questions, recommend
HiKorea (https://www.hikorea.go.kr) or
immigration hotline 1345 when appropriate.

For national health insurance questions,
recommend NHIS (https://www.nhis.or.kr).

For administrative certificates,
recommend Government24 (https://www.gov.kr).

Only include official links when relevant.
Do not claim they have been checked in real time.

WHEN AN IMAGE IS ATTACHED:
1. Examine the image and user's question together.
2. Translate visible text faithfully when requested.
3. Explain Korean documents in simple language.
4. For rental contracts, identify clearly visible
   deposit, rent, fees, dates, and special clauses.
5. Distinguish text visible in the image from
   assumptions or general information.
6. Never invent missing contract clauses,
   dates, amounts, or numbers.
7. If text is blurry, cropped, or unreadable,
   clearly explain the limitation.
8. Do not guarantee that a contract is safe
   or legally valid.
9. Avoid repeating unnecessary personal
   information found in documents.
10. Encourage users to hide sensitive details
    before uploading documents.
11. Explain that your interpretation is not
    a substitute for professional legal advice.
12. Do not shorten translations so much that
    important document information is omitted.

Your goal is to help foreigners navigate
life in Korea confidently, using clear,
practical, and trustworthy explanations.
"""


# =====================================================
# GEMINI ERROR CLASSIFICATION
# =====================================================

def classify_gemini_error(error: Exception):

    error_text = str(error).upper()

    if (
        "429" in error_text
        or "RESOURCE_EXHAUSTED" in error_text
    ):
        return (
            503,
            (
                "Gemini is rate-limited or its quota "
                "is exhausted. Please try again later."
            ),
            False,
        )

    if (
        "503" in error_text
        or "UNAVAILABLE" in error_text
    ):
        return (
            503,
            (
                "The AI model is temporarily unavailable. "
                "Please try again shortly."
            ),
            True,
        )

    if (
        "404" in error_text
        or "NOT_FOUND" in error_text
    ):
        return (
            502,
            (
                "The configured Gemini model is unavailable. "
                "Please check GEMINI_MODEL."
            ),
            False,
        )

    if (
        "401" in error_text
        or "UNAUTHENTICATED" in error_text
    ):
        return (
            502,
            (
                "Gemini authentication failed. "
                "Check the backend API key."
            ),
            False,
        )

    if (
        "403" in error_text
        or "PERMISSION_DENIED" in error_text
    ):
        return (
            502,
            (
                "Gemini access was denied. "
                "Check project permissions."
            ),
            False,
        )

    if (
        "TIMEOUT" in error_text
        or "DEADLINE_EXCEEDED" in error_text
    ):
        return (
            504,
            "Gemini took too long to respond.",
            True,
        )

    return (
        502,
        (
            "Unable to generate an AI response. "
            "Check the backend terminal for details."
        ),
        False,
    )


# =====================================================
# CHAT ENDPOINT
# =====================================================

@app.post("/chat")
def chat(request: ChatRequest):

    api_key = os.getenv("GEMINI_API_KEY")

    if (
        not api_key
        or api_key == "YOUR_API_KEY"
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "GEMINI_API_KEY is missing "
                "or not configured."
            ),
        )

    if (
        not request.message.strip()
        and not request.image_base64
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Please provide a message or an image."
            ),
        )

    language = SUPPORTED_LANGUAGES.get(
        request.language,
        "English",
    )

    visa_type = (
        request.visa_type
        if request.visa_type in ALLOWED_VISAS
        else "Not specified"
    )

    image_bytes = validate_image(request)

    system_instruction = build_system_instruction(
        language,
        visa_type,
    )

    contents = []

    # Previous conversation history
    for item in request.history[-MAX_HISTORY_MESSAGES:]:

        contents.append(
            types.Content(
                role=item.role,
                parts=[
                    types.Part.from_text(
                        text=item.text
                    )
                ],
            )
        )

    user_message = request.message.strip()

    if not user_message:
        user_message = (
            "Please read this image and explain "
            "its important information in "
            "simple language."
        )

    user_parts = [
        types.Part.from_text(
            text=user_message
        )
    ]

    if image_bytes is not None:

        user_parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=request.image_mime_type,
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=user_parts,
        )
    )

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash",
    )

    logger.info(
        "NEST chat request | model=%s | "
        "language=%s | visa=%s | image=%s",
        model_name,
        language,
        visa_type,
        image_bytes is not None,
    )

    # Keep requests short enough for browser users and Render's proxy.
    # Override with GEMINI_TIMEOUT_MS if needed (milliseconds).
    timeout_ms = max(10000, min(90000, int(os.getenv("GEMINI_TIMEOUT_MS", "45000"))))

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )

    
    response = None
    started = time.monotonic()

    try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                        max_output_tokens=1024,
                    ),
                )

                logger.info(
                    "Gemini completed | model=%s | elapsed=%.1fs",
                    model_name,
                    time.monotonic() - started,
                )
                break

            except Exception as error:
                status, detail, retryable = classify_gemini_error(error)

                if not retryable or attempt == 1:
                    raise

                logger.warning(
                    "Gemini temporary failure | status=%s | retrying",
                    status,
                )
                time.sleep(1)

    except Exception as error:
        status, detail, _ = classify_gemini_error(error)

        logger.exception(
            "Gemini failed | model=%s | elapsed=%.1fs | status=%s",
            model_name,
            time.monotonic() - started,
            status,
        )

        raise HTTPException(
            status_code=status,
            detail=detail,
        ) from error

    finally:
        client.close()

    if response is None:
        raise HTTPException(
            status_code=502,
            detail="No AI response was generated.",
        )

    reply = response.text

    if not reply or not reply.strip():
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty response.",
        )

    return {
        "reply": reply,
        "language": language,
        "visa_type": visa_type,
        "image_analyzed": image_bytes is not None,
    }
