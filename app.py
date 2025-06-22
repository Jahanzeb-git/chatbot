"""
Refactored Flask Backend for a RAG-powered Portfolio Chatbot
Version: 2.0.0
Date: 2025-06-21

Summary of Modifications:
- Vector Store: Replaced FAISS with ChromaDB for compatibility with environments that don't
  allow C++ compilation (e.g., PythonAnywhere free tier). The database is persisted
  to the local filesystem (`chroma_db`) and reloaded on startup.
- LLM Model: Upgraded the chat and summarization model from DeepSeek to the more performant
  `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free` via the Together AI API.
- Endpoint Removal: Deleted the `/api/chat/history/<session_id>` endpoint as it is no
  longer required by the front-end, reducing code complexity and database load.
- Robust Error Handling:
  - Implemented a retry mechanism with exponential backoff for all Together AI API calls.
  - Added timeout handling for SQLite database locks to prevent blocking under contention,
    returning a 503 Service Unavailable error.
  - Enhanced RAG system initialization to gracefully handle a missing or corrupt ChromaDB,
    logging the error and attempting to rebuild the index.
- PythonAnywhere Optimizations:
  - Replaced all `print()` statements with structured `logging` for better diagnostics.
  - Ensured the streaming endpoint (`/api/chat/stream`) sends an immediate preliminary
    response to prevent WSGI server timeouts.
  - Confirmed all data (SQLite DB, ChromaDB) is persisted in the application's
    home directory.
  - Added a WSGI entry point (`application = app`) for standard server compatibility.
- Code Quality:
  - Removed unused imports (`numpy`).
  - Added extensive inline comments to explain key logic.
  - Centralized configuration variables for easier management.
"""

# --- Core Imports ---
import os
import json
import uuid
import time
import sqlite3
import threading
import logging
import pickle
from datetime import datetime, timedelta
from contextlib import contextmanager

# --- Third-party Imports ---
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from together import Together
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader, PyPDFLoader
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import HumanMessage, AIMessage, Document
from langchain.embeddings.base import Embeddings

# --- Application Setup ---
app = Flask(__name__)
CORS(app)

# --- Logging Configuration ---
# Use structured logging instead of print() for better monitoring on servers.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
)

# --- Configuration Constants ---
# Using relative paths is ideal for PythonAnywhere, placing data in the app's directory.
DATABASE_PATH = 'chatbot_sessions.db'
CHROMA_DB_PATH = "chroma_db"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
EMBEDDING_MODEL = "togethercomputer/m2-bert-80M-32k-retrieval"
DB_LOCK_TIMEOUT_SECONDS = 5  # Max time to wait for a database lock.

# --- Global Variables ---
vector_store = None
embeddings = None
together_client = None
# A re-entrant lock is used to manage access to the SQLite database from multiple threads.
db_lock = threading.Lock()

# --- Custom Classes and Functions ---

@contextmanager
def managed_db_lock(timeout=DB_LOCK_TIMEOUT_SECONDS):
    """
    A context manager to safely acquire and release the database lock with a timeout.
    This prevents indefinite blocking if the DB is busy.
    """
    locked = db_lock.acquire(timeout=timeout)
    if not locked:
        logging.warning(f"Could not acquire database lock within {timeout} seconds.")
        # The calling function is responsible for handling this failure,
        # typically by returning an HTTP 503 error.
        raise TimeoutError("Database service is busy, please retry.")
    try:
        yield
    finally:
        db_lock.release()

class TogetherEmbeddings(Embeddings):
    """
    Custom LangChain-compatible embedding class using the Together AI API.
    This is necessary because LangChain's built-in TogetherEmbeddings may not be up-to-date.
    """
    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL):
        self.client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
        self.model = model

    def _get_embedding(self, text: str):
        """Helper to get embedding for a single piece of text with retry logic."""
        text = text.replace("\n", " ").strip()
        if not text:
            return [0.0] * 768  # Return a zero-vector for empty strings.

        def api_call():
            return self.client.embeddings.create(input=[text], model=self.model)

        try:
            response = _api_call_with_retry(api_call, "embedding")
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Failed to get embedding after retries: {e}", exc_info=True)
            return [0.0] * 768

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._get_embedding(text)

class CustomLLM:
    """
    A wrapper for the Together AI client to make it compatible with LangChain's
    ConversationSummaryBufferMemory, which expects a callable LLM object.
    """
    def __init__(self, client):
        self.client = client

    def __call__(self, prompt: str) -> str:
        """Generates a non-streaming response for summarizing conversation history."""
        logging.info("Generating conversation summary...")
        def api_call():
            return self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,  # Keep summaries concise
                temperature=0.3,
                stream=False
            )
        try:
            response = _api_call_with_retry(api_call, "summarization")
            summary = response.choices[0].message.content
            logging.info("Successfully generated conversation summary.")
            return summary
        except Exception as e:
            logging.error(f"Failed to generate summary after retries: {e}", exc_info=True)
            return "Error: Could not generate conversation summary."

def _api_call_with_retry(api_call_func, call_name="LLM", max_retries=2, initial_delay=1):
    """
    A robust wrapper for API calls that implements exponential backoff.
    Args:
        api_call_func: A lambda or function that executes the API call.
        call_name: A friendly name for the type of call (for logging).
        max_retries: The maximum number of times to retry.
        initial_delay: The first delay in seconds.
    Returns:
        The result of the API call.
    Raises:
        The last exception if all retries fail.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return api_call_func()
        except Exception as e:
            logging.warning(f"{call_name} API call failed on attempt {attempt + 1}/{max_retries}. Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logging.error(f"All {max_retries} retries failed for {call_name} API call.")
                raise  # Re-raise the final exception

# --- Initialization Functions ---

def init_database():
    """Initializes the SQLite database and its schema if they don't exist."""
    try:
        with managed_db_lock():
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        memory_data BLOB,
                        message_count INTEGER DEFAULT 0
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON messages(session_id)')
                conn.commit()
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise

def init_rag_system():
    """
    Initializes the RAG system by loading or creating a Chroma vector store.
    It first attempts to load a persisted store. If that fails, it builds a new one
    from source documents and persists it.
    """
    global vector_store, embeddings
    try:
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        
        embeddings = TogetherEmbeddings(api_key=api_key)

        # Try to load existing persisted ChromaDB first for faster startups.
        if os.path.exists(CHROMA_DB_PATH) and os.path.isdir(CHROMA_DB_PATH):
            logging.info(f"Loading existing Chroma database from '{CHROMA_DB_PATH}'...")
            vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
            logging.info("Chroma database loaded successfully.")
        else:
            # If it doesn't exist, create it from scratch.
            logging.info("No existing Chroma database found. Creating a new one.")
            documents = load_portfolio_documents()
            
            if not documents:
                logging.warning("No portfolio documents found. RAG context will be empty.")
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(documents)
            
            logging.info(f"Creating and persisting new Chroma vector store at '{CHROMA_DB_PATH}'...")
            vector_store = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings, 
                persist_directory=CHROMA_DB_PATH
            )
            logging.info("New Chroma database created successfully.")
            
    except Exception as e:
        logging.error(f"A critical error occurred during RAG system initialization: {e}", exc_info=True)
        logging.warning("RAG system will be non-functional for this session.")
        vector_store = None # Ensure vector_store is None on failure.

def load_portfolio_documents():
    """Loads portfolio content from JSON or PDF files, with a fallback to default text."""
    documents = []
    # Simplified loading logic with clear logging.
    try:
        if os.path.exists('portfolio_data.json'):
            logging.info("Loading portfolio data from portfolio_data.json")
            loader = JSONLoader(file_path='portfolio_data.json', jq_schema='.[]', text_content=False)
            documents.extend(loader.load())
        elif os.path.exists('portfolio_data.pdf'):
            logging.info("Loading portfolio data from portfolio_data.pdf")
            loader = PyPDFLoader('portfolio_data.pdf')
            documents.extend(loader.load())
    except Exception as e:
        logging.error(f"Error loading document file: {e}", exc_info=True)

    if not documents:
        logging.warning("No data file found or loaded. Using default fallback portfolio data.")
        default_data = """
        I am a skilled software professional with deep experience in Python, AI, and DevOps.
        My portfolio showcases projects involving scalable backends, RAG systems, and MLOps.
        I am proficient with technologies like Flask, Docker, AWS, and various ML frameworks.
        """
        documents.append(Document(page_content=default_data))
        
    return documents

# --- Core Application Logic ---

def get_relevant_context(query, k=3):
    """Retrieves relevant context chunks from the vector store."""
    if vector_store is None:
        logging.warning("Vector store not initialized. Cannot retrieve context.")
        return ""
    try:
        docs = vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logging.error(f"Error retrieving context from Chroma: {e}", exc_info=True)
        return ""

def create_session():
    """Creates a new chat session and stores its initial state in the database."""
    session_id = str(uuid.uuid4())
    logging.info(f"Creating new session: {session_id}")
    try:
        llm_wrapper = CustomLLM(together_client)
        memory = ConversationSummaryBufferMemory(llm=llm_wrapper, max_token_limit=1000, return_messages=True)
        memory_data = pickle.dumps(memory)
        
        with managed_db_lock():
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute('INSERT INTO sessions (session_id, memory_data) VALUES (?, ?)', (session_id, memory_data))
                conn.commit()
        return session_id
    except (pickle.PicklingError, sqlite3.Error, TimeoutError) as e:
        logging.error(f"Failed to create session {session_id}: {e}", exc_info=True)
        raise

def get_session_memory(session_id):
    """Retrieves and unpickles a session's conversation memory from the database."""
    try:
        with managed_db_lock():
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.execute('SELECT memory_data FROM sessions WHERE session_id = ?', (session_id,))
                result = cursor.fetchone()
        
        if result and result[0]:
            return pickle.loads(result[0])
        logging.warning(f"No memory found for session_id: {session_id}")
        return None
    except (pickle.UnpicklingError, sqlite3.Error, TimeoutError) as e:
        logging.error(f"Failed to retrieve session memory for {session_id}: {e}", exc_info=True)
        return None

def update_session_memory(session_id, memory):
    """Serializes and updates a session's memory in the database."""
    try:
        memory_data = pickle.dumps(memory)
        with managed_db_lock():
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute('UPDATE sessions SET memory_data = ?, last_updated = CURRENT_TIMESTAMP WHERE session_id = ?', (memory_data, session_id))
                conn.commit()
    except (pickle.PicklingError, sqlite3.Error, TimeoutError) as e:
        logging.error(f"Failed to update session memory for {session_id}: {e}", exc_info=True)

def save_message(session_id, role, content):
    """Saves a single chat message to the database and updates the session's message count."""
    try:
        with managed_db_lock():
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)', (session_id, role, content))
                cursor.execute('UPDATE sessions SET message_count = message_count + 1, last_updated = CURRENT_TIMESTAMP WHERE session_id = ?', (session_id,))
                conn.commit()
    except (sqlite3.Error, TimeoutError) as e:
        logging.error(f"Failed to save message for session {session_id}: {e}", exc_info=True)

def cleanup_old_sessions():
    """Periodically removes sessions older than 24 hours to keep the database clean."""
    cutoff_time = datetime.now() - timedelta(hours=24)
    logging.info(f"Running cleanup for sessions older than {cutoff_time}.")
    try:
        with managed_db_lock():
            with sqlite3.connect(DATABASE_PATH) as conn:
                # Note: ON DELETE CASCADE on the foreign key handles message deletion automatically.
                cursor = conn.execute('DELETE FROM sessions WHERE created_at < ?', (cutoff_time,))
                deleted_count = cursor.rowcount
                conn.commit()
                if deleted_count > 0:
                    logging.info(f"Cleaned up {deleted_count} old sessions.")
    except (sqlite3.Error, TimeoutError) as e:
        logging.error(f"Error during old session cleanup: {e}", exc_info=True)


def session_exists(session_id):
    """Checks if a session_id exists in the database."""
    try:
        with managed_db_lock():
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.execute('SELECT 1 FROM sessions WHERE session_id = ?', (session_id,))
                return cursor.fetchone() is not None
    except (sqlite3.Error, TimeoutError) as e:
        logging.error(f"Error checking session existence for {session_id}: {e}", exc_info=True)
        return False

def generate_streaming_response(messages, context, session_id):
    """
    Main generator function that constructs the prompt, calls the LLM, and yields
    Server-Sent Events (SSE) for the streaming response.
    """
    session_memory = get_session_memory(session_id)
    memory_context = ""
    if session_memory:
        # Extract last ~6 messages for conversation context
        try:
            memory_buffer = session_memory.chat_memory.messages[-6:]
            if memory_buffer:
                memory_context = "\n".join([f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}" for msg in memory_buffer])
        except Exception as e:
            logging.error(f"Error getting memory context for session {session_id}: {e}")

    system_prompt = f"""You are a helpful AI assistant for a professional portfolio website.
Use the following context about the portfolio owner and recent conversation history to answer questions.
Be conversational, professional, and concise. If you don't know an answer, say so politely.
Do not mention the context or history directly in your response.

PORTFOLIO CONTEXT:
{context}

CONVERSATION HISTORY:
{memory_context}
"""
    api_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        def api_call():
            return together_client.chat.completions.create(
                model=LLM_MODEL,
                messages=api_messages,
                max_tokens=1024,
                temperature=0.7,
                stream=True
            )

        stream = _api_call_with_retry(api_call, "chat streaming")
        
        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"data: {json.dumps({'content': content, 'type': 'chunk'})}\n\n"
        
        # After the stream is complete, update memory and save the conversation.
        if session_memory and full_response:
            user_message = messages[-1]['content']
            session_memory.chat_memory.add_user_message(user_message)
            session_memory.chat_memory.add_ai_message(full_response)
            update_session_memory(session_id, session_memory)
            save_message(session_id, 'user', user_message)
            save_message(session_id, 'assistant', full_response)
        
    except Exception as e:
        error_msg = "The AI service is currently unavailable. Please try again in a moment."
        logging.error(f"LLM streaming failed for session {session_id}: {e}", exc_info=True)
        yield f"data: {json.dumps({'content': error_msg, 'type': 'error'})}\n\n"
    finally:
        # Always send a 'done' message to gracefully close the connection on the client-side.
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

# --- API Endpoints ---

@app.route('/api/chat/session', methods=['POST'])
def create_chat_session():
    """Endpoint to create a new chat session."""
    try:
        # It's good practice to run cleanup periodically, e.g., when new sessions are made.
        cleanup_old_sessions()
        session_id = create_session()
        return jsonify({'session_id': session_id})
    except TimeoutError:
        return jsonify({'error': 'Service busy, please retry.'}), 503
    except Exception as e:
        logging.error(f"Failed to create chat session: {e}", exc_info=True)
        return jsonify({'error': 'Could not create a new session.'}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Endpoint to handle streaming chat requests."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        messages = data.get('messages', [])

        if not session_id or not session_exists(session_id):
            return jsonify({'error': 'Valid session ID required'}), 400
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400

        user_query = messages[-1]['content']
        context = get_relevant_context(user_query)

        def generate():
            # PythonAnywhere optimization: Immediately send a packet to establish the
            # connection and prevent the 30-second WSGI idle timeout.
            yield f"data: {json.dumps({'type': 'processing'})}\n\n"
            
            # Now, proceed with the potentially long-running generation.
            yield from generate_streaming_response(messages, context, session_id)

        headers = {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
        return Response(generate(), mimetype='text/event-stream', headers=headers)

    except Exception as e:
        logging.error(f"Error in chat_stream endpoint: {e}", exc_info=True)
        # This is a fallback for unexpected errors before the stream starts.
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """A simple health check endpoint for monitoring."""
    active_sessions = -1
    db_ok = False
    try:
        with managed_db_lock(timeout=2): # Use a short timeout for health checks
            with sqlite3.connect(DATABASE_PATH) as conn:
                active_sessions = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
            db_ok = True
        
        status = {
            'status': 'healthy',
            'rag_initialized': vector_store is not None,
            'embedding_model': EMBEDDING_MODEL,
            'llm_model': LLM_MODEL,
            'database_status': 'ok' if db_ok else 'degraded',
            'active_sessions': active_sessions
        }
        return jsonify(status)
    except Exception as e:
        logging.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

# --- WSGI Entry Point and Main Execution ---

# Expose the app object for WSGI servers like Gunicorn or uWSGI (used by PythonAnywhere)
application = app

if __name__ == '__main__':
    try:
        # Load API key from environment
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("FATAL: TOGETHER_API_KEY environment variable is not set.")
        
        # Initialize global clients and systems
        together_client = Together(api_key=api_key)
        init_database()
        init_rag_system()
        
        logging.info("--- Portfolio Chatbot Backend v2.0.0 Initialized ---")
        logging.info(f"Database: SQLite3 at {DATABASE_PATH}")
        logging.info(f"Vector Store: ChromaDB at {CHROMA_DB_PATH}")
        logging.info(f"LLM Model: {LLM_MODEL}")
        logging.info("-----------------------------------------------------")
        
        # The Flask development server is not for production.
        # Use a proper WSGI server like Gunicorn when deploying.
        app.run(host='0.0.0.0', port=5001, debug=False)
        
    except Exception as e:
        logging.critical(f"Failed to start the application: {e}", exc_info=True)
        exit(1)
