from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
import uuid
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from together import Together
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader, PyPDFLoader
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, Document
from langchain.embeddings.base import Embeddings
import os
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Global variables
vector_store = None
embeddings = None
together_client = None
embedding_client = None
db_lock = threading.Lock()

# Database configuration
DATABASE_PATH = 'chatbot_sessions.db'

class TogetherEmbeddings(Embeddings):
    """Custom embedding class for Together AI's m2-bert-80M-32k-retrieval model"""
    
    def __init__(self, api_key: str, model: str = "togethercomputer/m2-bert-80M-32k-retrieval"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1"
        )
        self.model = model
    
    def embed_documents(self, texts):
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text):
        """Embed a single query"""
        return self._get_embedding(text)
    
    def _get_embedding(self, text):
        """Get embedding for a single text"""
        try:
            # Clean text
            text = text.replace("\n", " ").strip()
            if not text:
                return [0.0] * 768  # Return zero vector for empty text
            
            response = self.client.embeddings.create(
                input=[text], 
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * 768  # Return zero vector on error

class CustomLLM:
    """Wrapper for Together AI to work with LangChain memory"""
    def __init__(self, client):
        self.client = client
    
    def __call__(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in CustomLLM: {e}")
            return "Error generating summary"

def init_database():
    """Initialize SQLite database for session management"""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                memory_data BLOB,
                message_count INTEGER DEFAULT 0
            )
        ''')
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Create index for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session_id ON messages(session_id)
        ''')
        
        conn.commit()
        print("Database initialized successfully")

def init_rag_system():
    """Initialize RAG system with portfolio data using Together AI embeddings"""
    global vector_store, embeddings
    
    try:
        # Initialize Together AI embeddings
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        
        embeddings = TogetherEmbeddings(api_key=api_key)
        
        # Load portfolio data
        documents = load_portfolio_documents()
        
        if not documents:
            print("No documents loaded, using default portfolio data")
            return
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store with Together AI embeddings
        print("Creating vector store with Together AI embeddings...")
        vector_store = FAISS.from_documents(splits, embeddings)
        
        # Save vector store for future use
        vector_store.save_local("faiss_index")
        print("RAG system initialized successfully with Together AI embeddings")
        
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        vector_store = None

def load_portfolio_documents():
    """Load portfolio documents from various sources"""
    documents = []
    
    try:
        # Try loading JSON file
        if os.path.exists('portfolio_data.json'):
            loader = JSONLoader(
                file_path='portfolio_data.json',
                jq_schema='.[]',
                text_content=False
            )
            documents.extend(loader.load())
            print("Loaded portfolio data from JSON file")
        
        # Try loading PDF file
        elif os.path.exists('portfolio_data.pdf'):
            loader = PyPDFLoader('portfolio_data.pdf')
            documents.extend(loader.load())
            print("Loaded portfolio data from PDF file")
        
        # Default portfolio data if no file exists
        else:
            default_data = """
            I am a Data Science professional with expertise in Python, Machine Learning, and AI.
            I have experience with Flask, React, and various data analysis tools.
            My portfolio includes projects in web development and machine learning applications.
            I specialize in building chatbots, RAG systems, and end-to-end ML pipelines.
            My technical skills include: Python, JavaScript, SQL, Docker, AWS, and various ML frameworks.
            """
            documents.append(Document(page_content=default_data))
            print("Using default portfolio data")
            
    except Exception as e:
        print(f"Error loading documents: {e}")
        # Fallback to default data
        default_data = """
        I am a Data Science professional with expertise in Python, Machine Learning, and AI.
        I have experience with Flask, React, and various data analysis tools.
        """
        documents.append(Document(page_content=default_data))
    
    return documents

def get_relevant_context(query, k=3):
    """Retrieve relevant context from vector store"""
    if vector_store is None:
        return ""
    
    try:
        docs = vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

def create_session():
    """Create new chat session in database"""
    session_id = str(uuid.uuid4())
    
    try:
        # Initialize conversation memory
        llm_wrapper = CustomLLM(together_client)
        memory = ConversationSummaryBufferMemory(
            llm=llm_wrapper,
            max_token_limit=1000,
            return_messages=True
        )
        
        # Serialize memory data
        memory_data = pickle.dumps(memory)
        
        with db_lock:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sessions (session_id, memory_data)
                    VALUES (?, ?)
                ''', (session_id, memory_data))
                conn.commit()
        
        print(f"Created new session: {session_id}")
        return session_id
        
    except Exception as e:
        print(f"Error creating session: {e}")
        raise

def get_session_memory(session_id):
    """Retrieve session memory from database"""
    try:
        with db_lock:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT memory_data FROM sessions WHERE session_id = ?
                ''', (session_id,))
                result = cursor.fetchone()
                
                if result:
                    return pickle.loads(result[0])
                return None
                
    except Exception as e:
        print(f"Error retrieving session memory: {e}")
        return None

def update_session_memory(session_id, memory):
    """Update session memory in database"""
    try:
        memory_data = pickle.dumps(memory)
        
        with db_lock:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE sessions 
                    SET memory_data = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', (memory_data, session_id))
                conn.commit()
                
    except Exception as e:
        print(f"Error updating session memory: {e}")

def save_message(session_id, role, content):
    """Save message to database"""
    try:
        with db_lock:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO messages (session_id, role, content)
                    VALUES (?, ?, ?)
                ''', (session_id, role, content))
                
                # Update message count
                cursor.execute('''
                    UPDATE sessions 
                    SET message_count = message_count + 1,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', (session_id,))
                
                conn.commit()
                
    except Exception as e:
        print(f"Error saving message: {e}")

def cleanup_old_sessions():
    """Remove sessions older than 24 hours"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with db_lock:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                
                # Delete old messages first (foreign key constraint)
                cursor.execute('''
                    DELETE FROM messages 
                    WHERE session_id IN (
                        SELECT session_id FROM sessions 
                        WHERE created_at < ?
                    )
                ''', (cutoff_time,))
                
                # Delete old sessions
                cursor.execute('''
                    DELETE FROM sessions WHERE created_at < ?
                ''', (cutoff_time,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    print(f"Cleaned up {deleted_count} old sessions")
                    
    except Exception as e:
        print(f"Error cleaning up sessions: {e}")

def session_exists(session_id):
    """Check if session exists in database"""
    try:
        with db_lock:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 1 FROM sessions WHERE session_id = ?
                ''', (session_id,))
                return cursor.fetchone() is not None
                
    except Exception as e:
        print(f"Error checking session existence: {e}")
        return False

def generate_streaming_response(messages, context, session_id):
    """Generate streaming response with context and memory"""
    
    # Get session memory
    session_memory = get_session_memory(session_id)
    
    # Get conversation history from memory
    memory_context = ""
    if session_memory:
        try:
            memory_buffer = session_memory.chat_memory.messages
            if memory_buffer:
                memory_context = "\n".join([
                    f"Human: {msg.content}" if isinstance(msg, HumanMessage) 
                    else f"Assistant: {msg.content}" 
                    for msg in memory_buffer[-6:]  # Last 3 exchanges
                ])
        except Exception as e:
            print(f"Error getting memory context: {e}")
    
    # Construct system prompt with context
    system_prompt = f"""You are a helpful AI assistant for a Data Science portfolio website. 
Use the following context about the portfolio owner to answer questions accurately:

PORTFOLIO CONTEXT:
{context}

CONVERSATION HISTORY:
{memory_context}

Guidelines:
- Answer questions about the portfolio owner's skills, projects, and background
- Be conversational and professional
- If you don't know something specific, say so politely
- Keep responses concise but informative
- Don't mention the context or conversation history directly
"""
    
    # Prepare messages for API
    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(messages)
    
    try:
        stream = together_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=api_messages,
            max_tokens=512,
            temperature=0.7,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                # Filter out thinking tags from DeepSeek
                if not ('<think>' in content or '</think>' in content):
                    full_response += content
                    yield f"data: {json.dumps({'content': content, 'type': 'chunk'})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        # Update memory and save messages
        if session_memory and full_response:
            try:
                user_message = messages[-1]['content']
                session_memory.chat_memory.add_user_message(user_message)
                session_memory.chat_memory.add_ai_message(full_response)
                
                # Update memory in database
                update_session_memory(session_id, session_memory)
                
                # Save messages to database
                save_message(session_id, 'user', user_message)
                save_message(session_id, 'assistant', full_response)
                
            except Exception as e:
                print(f"Error updating memory: {e}")
                
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(error_msg)
        yield f"data: {json.dumps({'content': error_msg, 'type': 'error'})}\n\n"

@app.route('/api/chat/session', methods=['POST'])
def create_chat_session():
    """Create new chat session"""
    try:
        cleanup_old_sessions()
        session_id = create_session()
        return jsonify({'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Handle streaming chat requests"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        messages = data.get('messages', [])
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        if not session_exists(session_id):
            return jsonify({'error': 'Invalid session'}), 400
        
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        # Get relevant context from RAG
        user_query = messages[-1]['content']
        context = get_relevant_context(user_query)
        
        # Generate streaming response
        def generate():
            yield from generate_streaming_response(messages, context, session_id)
        
        return Response(
            generate(),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream'
            }
        )
        
    except Exception as e:
        print(f"Error in chat_stream: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Get chat history for session"""
    try:
        if not session_exists(session_id):
            return jsonify({'error': 'Session not found'}), 404
        
        with db_lock:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                
                # Get session info
                cursor.execute('''
                    SELECT created_at, last_updated, message_count 
                    FROM sessions WHERE session_id = ?
                ''', (session_id,))
                session_info = cursor.fetchone()
                
                # Get messages
                cursor.execute('''
                    SELECT role, content, timestamp 
                    FROM messages 
                    WHERE session_id = ? 
                    ORDER BY timestamp ASC
                ''', (session_id,))
                messages = cursor.fetchall()
                
                return jsonify({
                    'session_id': session_id,
                    'created_at': session_info[0],
                    'last_updated': session_info[1],
                    'message_count': session_info[2],
                    'messages': [
                        {
                            'role': msg[0],
                            'content': msg[1],
                            'timestamp': msg[2]
                        } for msg in messages
                    ]
                })
                
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM sessions')
            active_sessions = cursor.fetchone()[0]
        
        return jsonify({
            'status': 'healthy',
            'rag_initialized': vector_store is not None,
            'embedding_model': 'togethercomputer/m2-bert-80M-32k-retrieval',
            'active_sessions': active_sessions,
            'database': 'sqlite3'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get application statistics"""
    try:
        with db_lock:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                
                # Get session stats
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN created_at > datetime('now', '-24 hours') THEN 1 END) as recent_sessions
                    FROM sessions
                ''')
                session_stats = cursor.fetchone()
                
                # Get message stats
                cursor.execute('''
                    SELECT COUNT(*) FROM messages
                ''')
                total_messages = cursor.fetchone()[0]
                
                return jsonify({
                    'total_sessions': session_stats[0],
                    'recent_sessions': session_stats[1],
                    'total_messages': total_messages,
                    'rag_status': 'active' if vector_store else 'inactive'
                })
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Initialize database
        init_database()
        
        # Initialize Together AI client
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        
        together_client = Together(api_key=api_key)
        
        # Initialize RAG system with Together AI embeddings
        init_rag_system()
        
        print("Backend v1.2 initialized successfully!")
        print("- Database: SQLite3")
        print("- Embeddings: Together AI m2-bert-80M-32k-retrieval")
        print("- LLM: DeepSeek-R1-Distill-Llama-70B")
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"Failed to initialize application: {e}")
        exit(1)