from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
from io import StringIO
import traceback
import os
from google import genai
from google.genai import types
import yt_dlp
import time
from pathlib import Path

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Code Interpreter Models
class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# YouTube Timestamp Models
class TimestampRequest(BaseModel):
    video_url: str
    topic: str


class TimestampResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


def execute_python_code(code: str) -> dict:
    """Execute Python code and return exact output."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception as e:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    """Use LLM with structured output to identify error line numbers."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{traceback_text}

Return the line number(s) where the error is located.
"""

    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER)
                    )
                },
                required=["error_lines"]
            )
        )
    )

    result = ErrorAnalysis.model_validate_json(response.text)
    return result.error_lines


def download_audio(video_url: str, output_path: str) -> str:
    """Download audio from YouTube video using yt-dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return f"{output_path}.mp3"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")


def find_timestamp_in_audio(audio_file_path: str, topic: str) -> str:
    """Upload audio to Gemini and find when topic is mentioned."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    # Upload audio file
    audio_file = client.files.upload(path=audio_file_path)
    
    # Wait for file to be processed
    max_wait = 300  # 5 minutes max
    wait_interval = 2
    elapsed = 0
    
    while audio_file.state.name == "PROCESSING" and elapsed < max_wait:
        time.sleep(wait_interval)
        audio_file = client.files.get(name=audio_file.name)
        elapsed += wait_interval
    
    if audio_file.state.name != "ACTIVE":
        raise HTTPException(status_code=500, detail="Audio file processing failed")
    
    # Create prompt for Gemini
    prompt = f"""
Listen to this audio and find the timestamp when the following topic or phrase is mentioned: "{topic}"

Return the timestamp in HH:MM:SS format (e.g., "00:05:47", "01:23:45").
If the topic is mentioned multiple times, return the FIRST occurrence.
If you cannot find the exact topic, return the timestamp of the most relevant related content.
"""
    
    # Use structured output to ensure HH:MM:SS format
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=[
            types.Part.from_uri(file_uri=audio_file.uri, mime_type=audio_file.mime_type),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "timestamp": types.Schema(type=types.Type.STRING)
                },
                required=["timestamp"]
            )
        )
    )
    
    # Delete the uploaded file from Gemini
    client.files.delete(name=audio_file.name)
    
    import json
    result = json.loads(response.text)
    return result["timestamp"]


@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):
    """Execute Python code and analyze errors with AI if needed."""
    execution_result = execute_python_code(request.code)
    
    if execution_result["success"]:
        return CodeResponse(
            error=[],
            result=execution_result["output"]
        )
    
    error_lines = analyze_error_with_ai(request.code, execution_result["output"])
    
    return CodeResponse(
        error=error_lines,
        result=execution_result["output"]
    )


@app.post("/ask", response_model=TimestampResponse)
async def find_timestamp(request: TimestampRequest):
    """Find timestamp of topic in YouTube video."""
    temp_dir = Path("/tmp") if os.path.exists("/tmp") else Path(".")
    audio_path = temp_dir / f"audio_{int(time.time())}"
    
    try:
        # Download audio
        audio_file = download_audio(request.video_url, str(audio_path))
        
        # Find timestamp using Gemini
        timestamp = find_timestamp_in_audio(audio_file, request.topic)
        
        return TimestampResponse(
            timestamp=timestamp,
            video_url=request.video_url,
            topic=request.topic
        )
    
    finally:
        # Clean up temporary files
        for ext in ['.mp3', '.webm', '.m4a', '.opus']:
            file_to_delete = Path(f"{audio_path}{ext}")
            if file_to_delete.exists():
                try:
                    file_to_delete.unlink()
                except:
                    pass


@app.get("/")
async def root():
    return {
        "message": "Code Interpreter & YouTube Timestamp Finder API",
        "endpoints": {
            "POST /code-interpreter": "Execute Python code with AI error analysis",
            "POST /ask": "Find timestamp of topic in YouTube video"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
