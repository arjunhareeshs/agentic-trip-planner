import io
import json
import logging
import os
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Response
from fastapi.responses import StreamingResponse
import httpx

from connectors.agent_runner import AgentRunnerController

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])

_controller = AgentRunnerController()

DEEPGRAM_STT_API_KEY = os.environ.get("DEEPGRAM_STT_API_KEY")
DEEPGRAM_TTS_API_KEY = os.environ.get("DEEPGRAM_TTS_API_KEY")

@router.post("/voice/process")
async def process_voice(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    user_id: str = Form("default_user"),
):
    """
    1. STT: Accept audio file, send to Deepgram STT.
    2. LLM: Send transpired text to AgentRunnerController.run_agent.
    3. TTS: Send LLM response text to Deepgram TTS.
    Returns: TTS MP3 audio stream with headers containing the LLM text.
    """
    if not DEEPGRAM_STT_API_KEY or not DEEPGRAM_TTS_API_KEY:
        raise HTTPException(
            status_code=500, detail="Deepgram API keys are not configured properly."
        )

    # 1. Speech-to-Text (STT) via Deepgram
    stt_text = ""
    try:
        audio_content = await audio.read()
        
        async with httpx.AsyncClient() as client:
            stt_response = await client.post(
                "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
                headers={
                    "Authorization": f"Token {DEEPGRAM_STT_API_KEY}",
                    "Content-Type": audio.content_type or "audio/webm",
                },
                content=audio_content,
                timeout=30.0,
            )
            stt_response.raise_for_status()
            stt_data = stt_response.json()
            stt_text = stt_data["results"]["channels"][0]["alternatives"][0]["transcript"]
            
            if not stt_text.strip():
                raise HTTPException(status_code=400, detail="Could not transcribe audio.")
    except Exception as e:
        logger.exception("Deepgram STT error")
        raise HTTPException(status_code=500, detail=f"STT Error: {str(e)}")

    logger.info(f"User voice transcribed as: {stt_text}")

    # 2. LLM Processing
    actual_session_id = session_id or str(uuid.uuid4())
    llm_response = ""
    try:
        result = await _controller.run_agent(
            message=stt_text,
            session_id=actual_session_id,
            user_id=user_id,
        )
        llm_response_full = result.get("response", "")
        # Deepgram TTS performs best without Markdown formatting.
        # We strip some basic markdown syntax.
        llm_response = llm_response_full.replace("*", "").replace("#", "").replace("`", "")
    except Exception as e:
        logger.exception("LLM error during voice interaction")
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

    if not llm_response.strip():
        llm_response = "I'm sorry, I couldn't generate a response."

    # 3. Text-to-Speech (TTS) via Deepgram Aura
    try:
        async with httpx.AsyncClient() as client:
            tts_response = await client.post(
                "https://api.deepgram.com/v1/speak?model=aura-asteria-en",
                headers={
                    "Authorization": f"Token {DEEPGRAM_TTS_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"text": llm_response},
                timeout=30.0,
            )
            tts_response.raise_for_status()
            
            # Streaming the MP3 back to the client
            audio_bytes = tts_response.content
            
    except Exception as e:
        logger.exception("Deepgram TTS error")
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")

    # Return the audio along with headers capturing the interaction context
    encoded_transcript = stt_text.replace('\n', ' ').encode('utf-8').decode('latin-1', 'ignore')
    encoded_llm = llm_response.replace('\n', ' ').encode('utf-8').decode('latin-1', 'ignore')
    
    headers = {
        "X-User-Transcript": "".join([c if ord(c) < 128 else "" for c in encoded_transcript]),
        "X-Bot-Response": "".join([c if ord(c) < 128 else "" for c in encoded_llm]),
        "X-Session-Id": actual_session_id,
        "Access-Control-Expose-Headers": "X-User-Transcript, X-Bot-Response, X-Session-Id",
    }
    
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers=headers,
    )
