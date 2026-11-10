import base64
import hmac
import io
import os
import re
import time
import textwrap
import unicodedata
import wave
from threading import Lock

import requests
from bidi.algorithm import get_base_level, get_display
from flask import Flask, Response, jsonify, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 320000
token = os.environ.get("BUDDY_TOKEN", "").encode()
api_key = os.environ.get("OPENAI_API_KEY", "")
api_url = "https://api.openai.com/v1"
headers = {"Authorization": f"Bearer {api_key}"}
sessions = {}
turn_lock = Lock()
last_turn = float("-inf")


@app.get("/health")
def health():
    ready = len(token) >= 24 and bool(api_key)
    return jsonify(ready=ready), 200 if ready else 503


@app.post("/v1/turn")
def turn():
    if len(token) < 24 or not api_key:
        return jsonify(error="Backend needs configuration"), 503
    if not hmac.compare_digest(request.headers.get("Authorization", "").encode(), b"Bearer " + token):
        return jsonify(error="Invalid device token"), 401

    session_id = request.headers.get("X-Buddy-Session", "")
    pcm = request.get_data()
    if (
        not re.fullmatch(r"[a-f0-9]{16}", session_id)
        or request.mimetype != "application/octet-stream"
        or not 6400 <= len(pcm) <= 320000
        or len(pcm) % 2
    ):
        return jsonify(error="Invalid audio or session"), 400

    if not turn_lock.acquire(blocking=False):
        return jsonify(error="Buddy is already processing a turn"), 429
    try:
        return process_turn(session_id, pcm)
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        response = exc.response if isinstance(exc, requests.RequestException) else None
        upstream_status = response.status_code if response is not None else None
        code = "unavailable"
        if response is not None:
            try:
                error = response.json()["error"]
                value = error.get("code") if isinstance(error, dict) else None
                if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
                    code = value
            except (ValueError, KeyError, TypeError):
                pass

        upstream_request = exc.request if isinstance(exc, requests.RequestException) else None
        stage = {
            f"{api_url}/audio/transcriptions": "transcription",
            f"{api_url}/chat/completions": "chat",
            f"{api_url}/audio/speech": "speech",
        }.get(getattr(upstream_request, "url", None), "unknown")
        app.logger.warning(
            "Voice service failed (%s, status %s, stage %s, code %s)",
            type(exc).__name__, upstream_status, stage, code,
        )
        return jsonify(error="Voice provider request failed"), 502
    finally:
        turn_lock.release()


def process_turn(session_id, pcm):
    global last_turn
    now = time.monotonic()
    if now - last_turn < 2:
        return jsonify(error="Please wait before trying again"), 429
    last_turn = now
    for key, (updated, _) in list(sessions.items()):
        if now - updated > 900:
            del sessions[key]

    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)

    transcription = requests.post(
        f"{api_url}/audio/transcriptions", headers=headers, timeout=(5, 25),
        data={"model": os.getenv("TRANSCRIBE_MODEL", "gpt-transcribe")},
        files={"file": ("question.wav", audio.getvalue(), "audio/wav")},
    )
    transcription.raise_for_status()
    question = transcription.json()["text"].strip()[:1500]
    if not question:
        return "", 204

    _, history = sessions.get(session_id, (now, []))
    messages = [
        {"role": "system", "content": (
            "You are Buddy, Ofer's desk companion. Reply in the language of his latest question, "
            "including Hebrew when he speaks Hebrew. Speak naturally in at most two short "
            "sentences. Use plain text suitable for reading aloud, with no markdown."
        )},
        *history,
        {"role": "user", "content": question},
    ]
    reply = requests.post(
        f"{api_url}/chat/completions", headers=headers, timeout=(5, 25),
        json={"model": os.getenv("CHAT_MODEL", "gpt-6-luna"), "messages": messages,
              "max_completion_tokens": 160, "reasoning_effort": "none", "store": False},
    )
    reply.raise_for_status()
    answer = reply.json()["choices"][0]["message"]["content"].strip()
    if not answer:
        return "", 204

    with requests.post(
        f"{api_url}/audio/speech", headers=headers, timeout=(5, 25), stream=True,
        json={"model": os.getenv("SPEECH_MODEL", "gpt-4o-mini-tts"),
              "voice": os.getenv("SPEECH_VOICE", "marin"), "input": answer, "response_format": "pcm"},
    ) as speech:
        speech.raise_for_status()
        spoken = bytearray()
        for chunk in speech.iter_content(16384):
            spoken.extend(chunk)
            if len(spoken) > 1440000:
                return jsonify(error="Spoken reply exceeded 30 seconds"), 502
    if not spoken or len(spoken) % 2:
        return jsonify(error="Invalid spoken reply"), 502

    if session_id not in sessions and len(sessions) >= 32:
        del sessions[next(iter(sessions))]
    sessions[session_id] = (time.monotonic(), [
        *history, messages[-1], {"role": "assistant", "content": answer}
    ][-6:])
    response = Response(bytes(spoken), mimetype="application/octet-stream")
    if request.headers.get("X-Buddy-Captions") == "1":
        response.headers["X-Buddy-Text"] = encode_captions(answer)
    return response


def encode_captions(answer):
    # Match the device's single-cell ASCII and unpointed Hebrew glyphs.
    text = unicodedata.normalize("NFKD", answer).translate(str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
        "־": "-", "״": '"', "׳": "'", "×": "x",
    }))
    text = "".join(
        char if (" " <= char <= "~" or "א" <= char <= "ת") else
        " " if char.isspace() else "?"
        for char in text
        if unicodedata.category(char) not in {"Mn", "Me", "Cf"}
    )
    direction = "R" if get_base_level(text) else "L"
    lines = textwrap.wrap(text, width=26, max_lines=32, placeholder="...")
    visual = "\n".join(direction + get_display(line, base_dir=direction) for line in lines)
    return base64.b64encode(visual.encode("utf-8")).decode("ascii")
