"""
Vikaspuri House Construction Assistant — Backend API
====================================================
Flask app that proxies Google Sheets and OpenAI so no secrets ever
reach the browser.  All keys live in backend/.env.

Routes
------
GET  /              → Serve the SPA (templates/index.html)
GET  /api/status    → Health check + sheet connection status
POST /api/chat      → Chat with GPT-4 using live sheet context
POST /api/refresh   → Force-refresh the Google Sheets cache
"""

import logging
import os
import re
import threading
import time
from collections import defaultdict

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import openai

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()  # reads backend/.env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("vikaspuri")

app = Flask(__name__)

# CORS only needed when running the SPA from a different origin (e.g. file://
# during development).  Flask serves index.html on "/", so same-origin by
# default — this is a safety net.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Configuration (from .env)
# ---------------------------------------------------------------------------

OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL          = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GOOGLE_SHEETS_API_KEY = os.getenv("GOOGLE_SHEETS_API_KEY", "")
GOOGLE_SHEET_ID       = os.getenv("GOOGLE_SHEET_ID", "")

SHEET_TABS            = ["Owners", "Transactions", "Milestones", "Land", "FloorDetails", "Documents"]
SHEET_CACHE_TTL       = int(os.getenv("SHEET_CACHE_TTL_SECONDS", "300"))  # 5 min default

RATE_LIMIT_MAX        = int(os.getenv("RATE_LIMIT_MAX", "20"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "900"))  # 15 min

MAX_MESSAGE_LENGTH    = 1000  # characters — validated at boundary

# ---------------------------------------------------------------------------
# Google Sheets cache
# ---------------------------------------------------------------------------

_cache: dict = {"data": {}, "last_loaded": 0.0, "errors": []}
_cache_lock = threading.Lock()


def _fetch_sheets() -> tuple:
    """Fetch every tab from the Google Sheets API.  Returns (data, errors)."""
    if not GOOGLE_SHEETS_API_KEY or not GOOGLE_SHEET_ID:
        return {}, ["GOOGLE_SHEETS_API_KEY or GOOGLE_SHEET_ID not set in .env"]

    data: dict = {}
    errors: list = []
    base_url = "https://sheets.googleapis.com/v4/spreadsheets"

    for tab in SHEET_TABS:
        url = (
            f"{base_url}/{GOOGLE_SHEET_ID}/values/"
            f"{requests.utils.quote(tab)}?key={GOOGLE_SHEETS_API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=10)
            if resp.ok:
                values = resp.json().get("values", [])
                data[tab] = values
                logger.info("Sheet tab '%s' loaded — %d rows", tab, len(values))
            else:
                msg = f"Tab '{tab}': HTTP {resp.status_code}"
                errors.append(msg)
                logger.warning(msg)
        except requests.RequestException as exc:
            msg = f"Tab '{tab}': {exc}"
            errors.append(msg)
            logger.warning(msg)

    return data, errors


def get_sheet_data(force: bool = False) -> tuple:
    """Return cached sheet data, refreshing when stale or forced."""
    now = time.time()
    with _cache_lock:
        if (
            not force
            and _cache["data"]
            and (now - _cache["last_loaded"]) < SHEET_CACHE_TTL
        ):
            return _cache["data"], _cache["errors"]

    data, errors = _fetch_sheets()

    with _cache_lock:
        if data:
            _cache["data"] = data
            _cache["last_loaded"] = time.time()
        _cache["errors"] = errors

    return _cache["data"], _cache["errors"]


# Warm the cache in a background thread on startup so the first request is fast
threading.Thread(target=lambda: get_sheet_data(force=True), daemon=True).start()

# ---------------------------------------------------------------------------
# Sheet → system-prompt context builder
# ---------------------------------------------------------------------------

_LINK_PATTERN = re.compile(
    r"https?://\S+"
    r"|docs\.google\.com/\S+"
    r"|drive\.google\.com/\S+"
)


def _strip_links(value: object) -> str:
    """Replace any URLs/doc links with a placeholder."""
    return _LINK_PATTERN.sub("[document link hidden]", str(value or ""))


def _row_to_text(headers: list, row: list) -> str:
    return " | ".join(
        f"{headers[j] if j < len(headers) else f'Col{j + 1}'}: "
        f"{_strip_links(row[j] if j < len(row) else '-')}"
        for j in range(len(headers))
    )


def _documents_context(rows: list) -> str:
    if not rows:
        return ""
    headers = rows[0]
    lines = ["\n=== Documents Summary (links hidden) ==="]
    for row in rows[1:]:
        if not any(str(c or "").strip() for c in row):
            continue
        lines.append(_row_to_text(headers, row))
    lines.append(
        "Instruction: Never display raw document links to the user. "
        "Summarise document details. If the document text is not loaded, say: "
        "'The document is listed in records but the full text is not available in chat.'"
    )
    return "\n".join(lines) + "\n"


def build_sheet_context(sheet_data: dict) -> str:
    if not sheet_data:
        return "No sheet data is available. The Google Sheets connection may not be configured."

    parts = [
        "PROJECT: Vikaspuri House — 4-floor apartment building, Hyderabad, India. "
        "Co-owners: 4 friends, one per floor. "
        "Data is loaded live from Google Sheets."
    ]

    for tab, rows in sheet_data.items():
        if not rows:
            continue
        if tab.lower() == "documents":
            parts.append(_documents_context(rows))
            continue

        headers = rows[0]
        section_lines = [f"\n=== {tab} ==="]
        for row in rows[1:]:
            if not any(str(c or "").strip() for c in row):
                continue
            section_lines.append(_row_to_text(headers, row))
        parts.append("\n".join(section_lines))

    return "\n".join(parts)


def build_system_prompt(sheet_context: str, lang: str) -> str:
    lang_instruction = (
        "The user wrote in Telugu. Respond fully in Telugu script."
        if lang == "te"
        else "Respond in English unless the user writes in Telugu."
    )
    return (
        "You are a helpful assistant for the Vikaspuri House construction project "
        "in Hyderabad, India.\n"
        "This is a 4-floor apartment building where 4 friends are co-owners — "
        "each friend owns one floor.\n"
        "The project covers everything from land purchase to full construction completion.\n\n"
        "Here is all the current project data from Google Sheets:\n"
        f"{sheet_context}\n\n"
        "Instructions:\n"
        "- Answer ONLY based on the sheet data above. If specific data is missing, say clearly: "
        "\"This information is not in the records yet.\"\n"
        "- Use INR format with ₹ and Indian comma grouping. Use lakhs/crores where helpful.\n"
        "- Be concise by default: 4–6 bullets or one short paragraph.\n"
        "- For sale deed, land purchase, or document questions, give 6–10 short bullets.\n"
        "- Do NOT show raw Google Drive, Google Docs, or any document URLs in your response.\n"
        "- If a document link exists but document text is not loaded, say: "
        "\"The document is listed in records, but the full text is not available in chat.\"\n"
        "- Present tabular data clearly with labels.\n"
        f"- {lang_instruction}\n"
        "- Do not invent or assume facts not present in the Sheet data."
    )


# ---------------------------------------------------------------------------
# Per-IP rate limiter  (in-memory; fine for a single-server deployment)
# ---------------------------------------------------------------------------

_rate_lock = threading.Lock()
_rate_store: dict = defaultdict(list)  # ip → [unix_ts, ...]


def check_and_record(ip: str) -> tuple:
    """
    Returns (allowed, remaining, cooldown_ms).
    Records the request timestamp when allowed.
    """
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SEC

    with _rate_lock:
        timestamps = [t for t in _rate_store[ip] if t > cutoff]

        if len(timestamps) >= RATE_LIMIT_MAX:
            oldest = min(timestamps)
            cooldown_ms = int((oldest + RATE_LIMIT_WINDOW_SEC - now) * 1000)
            _rate_store[ip] = timestamps
            return False, 0, max(0, cooldown_ms)

        timestamps.append(now)
        _rate_store[ip] = timestamps
        return True, RATE_LIMIT_MAX - len(timestamps), 0


def _client_ip() -> str:
    """Best-effort real client IP, respecting X-Forwarded-For."""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page application."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Health check — also warms the cache if needed."""
    sheet_data, errors = get_sheet_data()
    return jsonify({
        "status": "ok",
        "openai_configured":  bool(OPENAI_API_KEY),
        "sheets_configured":  bool(GOOGLE_SHEETS_API_KEY and GOOGLE_SHEET_ID),
        "tabs_loaded":        len(sheet_data),
        "tabs":               list(sheet_data.keys()),
        "errors":             errors,
    })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Force-refresh the Google Sheets cache and return new status."""
    sheet_data, errors = get_sheet_data(force=True)
    return jsonify({
        "tabs_loaded": len(sheet_data),
        "tabs":        list(sheet_data.keys()),
        "errors":      errors,
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Accept a user message, fetch live sheet context, call OpenAI, return reply.

    Request JSON:
        { "message": "...", "lang": "en" | "te" }

    Success response:
        { "reply": "...", "remaining": <int> }

    Rate-limit response (HTTP 429):
        { "error": "rate_limited", "cooldown_ms": <int>, "remaining": 0 }
    """
    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key is not configured on the server. "
                                 "Please set OPENAI_API_KEY in backend/.env."}), 503

    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    lang    = str(body.get("lang",    "en")).strip()

    # ------------------------------------------------------------------
    # Input validation (system boundary)
    # ------------------------------------------------------------------
    if not message:
        return jsonify({"error": "message is required"}), 400

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({
            "error": f"Message is too long (max {MAX_MESSAGE_LENGTH} characters)."
        }), 400

    if lang not in ("en", "te"):
        lang = "en"

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    ip = _client_ip()
    allowed, remaining, cooldown_ms = check_and_record(ip)

    if not allowed:
        logger.info("Rate limit hit for IP %s", ip)
        return jsonify({
            "error":       "rate_limited",
            "cooldown_ms": cooldown_ms,
            "remaining":   0,
        }), 429

    # ------------------------------------------------------------------
    # Build context and system prompt (stays on the server)
    # ------------------------------------------------------------------
    sheet_data, _   = get_sheet_data()
    sheet_context   = build_sheet_context(sheet_data)
    system_prompt   = build_system_prompt(sheet_context, lang)

    # ------------------------------------------------------------------
    # Call OpenAI
    # ------------------------------------------------------------------
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": message},
            ],
            max_tokens=600,
            temperature=0.1,
        )
        reply = (
            completion.choices[0].message.content
            or "Sorry, I did not get a response. Please try again."
        )
        logger.info("Reply sent to %s  (remaining prompts: %d)", ip, remaining)
        return jsonify({"reply": reply, "remaining": remaining})

    except openai.AuthenticationError:
        logger.error("OpenAI authentication failed — check OPENAI_API_KEY in .env")
        return jsonify({"error": "OpenAI authentication failed. "
                                 "Check OPENAI_API_KEY in backend/.env."}), 503

    except openai.RateLimitError:
        logger.error("OpenAI rate limit / quota exceeded")
        return jsonify({"error": "OpenAI rate limit reached. "
                                 "Please try again in a moment."}), 503

    except openai.APIError as exc:
        logger.error("OpenAI API error: %s", exc)
        return jsonify({"error": f"OpenAI error: {exc.message}"}), 500

    except Exception:
        logger.exception("Unexpected error in /api/chat")
        return jsonify({"error": "An unexpected server error occurred. "
                                 "Please try again."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port  = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting Vikaspuri House backend on http://localhost:%d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
