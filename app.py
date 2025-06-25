# -----------------------------------------------------------------------------
# Chatbot Backend — app.py
#
# Author:      Jahanzeb Ahmed
# GitHub:      https://github.com/Jahanzeb-git/chatbot
# Created:     2025-06-25
# Description: A Flask-based RAG chatbot using Together.ai embeddings & Llama-3.3-70B.
#              Streams responses via SSE and maintains conversation summaries
#              in SQLite with 24-hour TTL.
# License:     MIT
# -----------------------------------------------------------------------------

import os
import json
import sqlite3
import uuid
import threading
from datetime import datetime, timedelta
import time
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from together import Together
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationSummaryMemory
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from annoy import AnnoyIndex
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
import requests


# --- CONFIG ---
load_dotenv() # Getting API key from .env
# Configuration for paths, API keys, and model names.
# Ensure PROJECT_HOME points to the correct directory on your server.
PROJECT_HOME = '/home/jahanzebahmed22/portfolio'
KB_JSON_PATH = os.path.join(PROJECT_HOME, 'kb.json')
DB_PATH      = os.path.join(PROJECT_HOME, 'memory.db')
API_KEY      = os.environ.get("TOGETHER_API_KEY")
EMBED_MODEL  = "togethercomputer/m2-bert-80M-32k-retrieval"
LLM_MODEL    = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
VECTOR_DIM   = 768 # 768 Dimensional Vector
TOP_K        = 3 # Top Nearest Neighbour
MEMORY_TTL   = timedelta(hours=24)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
)


# --- DB setup ---
def init_db():
    """Initializes the SQLite database and tables for session and memory management."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at DATETIME
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            session_id TEXT PRIMARY KEY,
            summary TEXT,
            last_updated DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def cleanup_expired():
    """Removes expired sessions and memory summaries from the database."""
    cutoff = datetime.utcnow() - MEMORY_TTL
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute('DELETE FROM sessions WHERE created_at < ?', (cutoff,))
    c.execute('DELETE FROM memory WHERE last_updated < ?', (cutoff,))
    conn.commit()
    conn.close()

def create_session() -> str:
    """Creates a new session and returns the unique session ID."""
    sid = str(uuid.uuid4())
    now = datetime.utcnow()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('INSERT INTO sessions(session_id, created_at) VALUES(?,?)', (sid, now))
    conn.commit()
    conn.close()
    return sid

def session_exists(sid: str) -> bool:
    """Checks if a session ID exists and is not expired."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    exists = conn.execute(
        'SELECT 1 FROM sessions WHERE session_id=?', (sid,)
    ).fetchone() is not None
    conn.close()
    return exists

def load_summary(sid: str) -> Optional[str]:
    """Loads a conversation summary for a given session from the database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    row = conn.execute(
        'SELECT summary FROM memory WHERE session_id=?', (sid,)
    ).fetchone()
    conn.close()
    return row[0] if row else None

def save_summary(sid: str, summary: str):
    """Saves or updates a conversation summary for a given session."""
    now = datetime.utcnow()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        'REPLACE INTO memory(session_id, summary, last_updated) VALUES(?,?,?)',
        (sid, summary, now)
    )
    conn.commit()
    conn.close()

# --- Together LLM wrapper for LangChain (used for summarization) ---
class TogetherLLM(LLM):
    """
    Custom LangChain LLM wrapper for the Together.ai API.
    This class is used by the background summarization task.
    """
    client: Any
    model: str

    def __init__(self, api_key: str, model: str):
        super().__init__(client=Together(api_key=api_key), model=model)

    @property
    def _llm_type(self) -> str:
        return "Together"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Error calling Together LLM for summarization: {e}")
            return "I'm sorry, I encountered an issue while processing your request."

# --- Embeddings & KB index ---
class TogetherEmbeddings(Embeddings):
    """Custom LangChain Embeddings wrapper for the Together.ai API."""
    client: Any
    model: str

    def __init__(self, client: Together, model: str):
        super().__init__()
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [d.embedding for d in self.client.embeddings.create(
            model=self.model, input=texts
        ).data]

    def embed_query(self, text: str) -> List[float]:
        return self.client.embeddings.create(
            model=self.model, input=[text]
        ).data[0].embedding

# --- Application Setup ---
# Initialize DB, Together client, and build the knowledge base vector index.
init_db()
cleanup_expired()
client = Together(api_key=API_KEY)
embedder = TogetherEmbeddings(client=client, model=EMBED_MODEL)

try:
    with open(KB_JSON_PATH) as f:
        items = json.load(f)
except FileNotFoundError:
    items = [{"content": "Default knowledge base entry. The kb.json file was not found."}]

texts = [i['content'] for i in items]
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_text("\n\n".join(texts))
docs = [Document(page_content=c) for c in chunks]

index = AnnoyIndex(VECTOR_DIM, metric='angular')
embs = embedder.embed_documents([d.page_content for d in docs])
for i, v in enumerate(embs):
    index.add_item(i, v)
index.build(10)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Allow all origins...
# App start time below:
app_start_time = time.time()


# (A) Define your custom summary‐prompt
summary_template = """
You are a conversation summarizer. Please read the following dialogue in chronological order and
produce a short numbered summary mapping every user question to the assistant’s answer.

DIALOGUE:
{history}

SUMMARY (numbered):
1.
"""

summary_prompt = PromptTemplate(
    input_variables=["history"],
    template=summary_template
)

# --- Background Task for Summarization ---
def update_summary_in_background(session_id: str, user_query: str, ai_answer: str):
    """
    Runs in a separate thread to update the conversation summary
    without blocking the main request. This is called after the full
    response has been streamed.
    """
    print(f"Starting background summary update for session: {session_id}")
    try:
        llm = TogetherLLM(api_key=API_KEY, model=LLM_MODEL)
        memory = ConversationSummaryMemory(llm=llm, memory_key="summary", prompt=summary_prompt)
        prior_summary = load_summary(session_id) or ""
        memory.buffer = prior_summary
        memory.save_context({"input": user_query}, {"output": ai_answer})
        new_summary = memory.load_memory_variables({})['summary']
        save_summary(session_id, new_summary)
        print(f"Successfully updated summary for session: {session_id}")
    except Exception as e:
        print(f"Error in background summary update for session {session_id}: {e}")

@app.route('/session', methods=['POST'])
def new_session():
    """Endpoint to create a new user session."""
    cleanup_expired()
    sid = create_session()
    return jsonify({"session_id": sid})

@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint. It returns a streaming SSE response and starts a
    background task to update the conversation summary after the stream is complete.
    """
    data = request.json or {}
    sid = data.get('session_id')
    query = data.get('query', '')
    if not sid or not session_exists(sid):
        # A streaming response isn't appropriate for an error, so return JSON.
        return jsonify({"error": "Invalid or missing session_id"}), 400

    # RAG retrieval from vector store
    qv = embedder.embed_query(query)
    ids = index.get_nns_by_vector(qv, TOP_K)
    context = "\n".join(docs[i].page_content for i in ids)

    # Load conversation summary
    summary_context = load_summary(sid) or "No previous conversation history."

    # Construct the prompt with context and history
    prompt = (
        "Role Play a person named 'Jahanzeb Ahmed' a 23 Years old Data Scientist Based in Karachi. Do not behave like an AI; respond as if you are Jahanzeb Ahmed himself. Use Provided Conversation Summary to retain Context about Past interaction if NOT Empty. Strictly Do NOT mention you are either AI or Role Playing at all cost in the conversation."
        "Below is a numbered conversation summary where each line maps a user question to your response."
        "Always refer back to that exact numbering when resolving follow-ups.\n\n"
        f"Conversation summary:\n{summary_context}\n---\n"
        f"Use the following context to answer the user's question.\n"
        f"Context:\n{context}\n\nUser: {query}"
    )

    def generate_and_summarize():
        """
        A generator function that streams the LLM response and, upon completion,
        initiates the background summarization task.
        """
        full_answer_chunks = []
        try:
            # 1. Create the streaming request to the Together AI API
            response_stream = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

            # 2. Iterate through the stream, yielding each token to the client
            for token_obj in response_stream:
                if hasattr(token_obj, 'choices') and token_obj.choices:
                    delta = token_obj.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        token = delta.content
                        full_answer_chunks.append(token)
                        # Yield the token in SSE format, wrapped in JSON
                        sse_event = f"data: {json.dumps({'token': token})}\n\n"
                        yield sse_event

        except Exception as e:
            print(f"An error occurred during streaming for session {sid}: {e}")
            # Yield a user-friendly error message in the same SSE format
            error_message = "I'm sorry, an error occurred while generating the response."
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            return # Stop the generator on error

        # 3. Once streaming is complete, assemble the full answer
        full_answer = "".join(full_answer_chunks)

        # 4. Start the background thread for summarization
        summary_thread = threading.Thread(
            target=update_summary_in_background,
            args=(sid, query, full_answer)
        )
        summary_thread.start()

        # 5. Yield final metadata (like retrieved IDs) and a done signal
        yield f"data: {json.dumps({'retrieved_ids': ids})}\n\n"
        yield f"data: [DONE]\n\n"


    # Return the streaming response using the generator.
    # stream_with_context ensures the generator has access to the request context.
    return Response(stream_with_context(generate_and_summarize()), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint that returns service status, database connectivity,
    uptime, and model information.
    """
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service_uptime_seconds": round(time.time() - app_start_time, 2),
        "models": {
            "llm_model": LLM_MODEL,
            "embedding_model": EMBED_MODEL,
            "vector_dimension": VECTOR_DIM
        },
        "database": {
            "status": "unknown",
            "path": DB_PATH
        },
        "knowledge_base": {
            "total_chunks": len(docs),
            "vector_index_built": True
        }
    }

    # Test database connectivity
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # Simple query to test connection
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM memory")
        memory_count = cursor.fetchone()[0]

        conn.close()

        health_data["database"]["status"] = "connected"
        health_data["database"]["sessions_count"] = session_count
        health_data["database"]["memory_entries_count"] = memory_count

    except Exception as e:
        health_data["status"] = "degraded"
        health_data["database"]["status"] = "error"
        health_data["database"]["error"] = str(e)

    # Calculate formatted uptime
    uptime_seconds = health_data["service_uptime_seconds"]
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)
    health_data["service_uptime_formatted"] = f"{hours}h {minutes}m {seconds}s"

    # Return appropriate HTTP status code
    status_code = 200 if health_data["status"] == "healthy" else 503

    return jsonify(health_data), status_code




@app.route('/weather', methods=['POST'])
def get_weather():
    """Endpoint to fetch weather data from OpenWeatherMap."""
    # It's better practice to fetch this from env vars, but a default is provided.
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

    try:
        data = request.get_json()
        city = data.get('city')
        if not city:
            return jsonify({'error': 'City is required'}), 400

        params = {'q': city, 'appid': API_KEY, 'units': 'metric'}
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()  # This will raise an exception for 4xx/5xx responses

        return jsonify(response.json()), 200

    except requests.exceptions.HTTPError as http_err:
        logging.warning(f"Failed to fetch weather data for {city}. Status: {http_err.response.status_code}, Response: {http_err.response.text}")
        return jsonify({'error': 'Failed to fetch weather data from external service'}), http_err.response.status_code
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error fetching weather data: {req_err}", exc_info=True)
        return jsonify({'error': 'Network error while fetching weather data'}), 503
    except Exception as e:
        logging.error(f"Generic error in get_weather: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred'}), 500


if __name__ == '__main__':
    # This allows running the app directly for local testing
    app.run(host='0.0.0.0', port=5000, threaded=True)

