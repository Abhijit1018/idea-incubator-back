import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from models import db, User, CatalogEntry, ChatMessage, CatalogEmbedding
from datetime import datetime
import uuid
import requests
import json
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import threading
import time
from datetime import timedelta

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    Pinecone = None
    ServerlessSpec = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

load_dotenv()

# Vector services are optional. Keep disabled on small instances by default.
embedding_model = None
pc = None
index = None
ENABLE_VECTOR_SEARCH = os.getenv('ENABLE_VECTOR_SEARCH', 'false').lower() == 'true'

if ENABLE_VECTOR_SEARCH and SentenceTransformer:
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Embedding model initialization failed: {e}")

pinecone_api_key = os.getenv('PINECONE_API_KEY')
if ENABLE_VECTOR_SEARCH and pinecone_api_key and Pinecone and ServerlessSpec:
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = os.getenv('PINECONE_INDEX_NAME', 'mindinspo-catalogs')
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=os.getenv('PINECONE_CLOUD', 'aws'),
                    region=os.getenv('PINECONE_REGION', 'us-east-1')
                )
            )
        index = pc.Index(index_name)
    except Exception as e:
        print(f"Pinecone initialization failed: {e}")

app = Flask(__name__)
CORS(app)

# Use SQLite for local development ease, but can switch to Postgres via DATABASE_URL
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///incubator.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

db.init_app(app)

# Test database connection
def test_db_connection():
    try:
        db.session.execute(db.text('SELECT 1'))
        print("Database connection OK")
    except Exception as e:
        print(f"Database connection error: {e}")

N8N_WEBHOOK_URL = os.getenv('N8N_WEBHOOK_URL', 'http://localhost:5678/webhook/idea-incubator')
N8N_CHAT_WEBHOOK_URL = os.getenv('N8N_CHAT_WEBHOOK_URL', 'http://localhost:5678/webhook/chat-message')
BACKEND_BASE_URL = os.getenv('BACKEND_BASE_URL', 'http://localhost:5000').rstrip('/')

def init_db():
    """Create tables, retrying up to 5 times with back-off to survive slow cold starts."""
    import time as _time
    for attempt in range(1, 6):
        try:
            db.create_all()
            test_db_connection()
            print("Database initialized OK")
            return
        except Exception as e:
            print(f"DB init attempt {attempt}/5 failed: {e}")
            if attempt < 5:
                _time.sleep(attempt * 3)  # 3s, 6s, 9s, 12s
    print("WARNING: DB init failed after 5 attempts. Tables will be created on first request.")

with app.app_context():
    init_db()

def background_retry_job(app):
    """
    Background job that periodically checks for pending entries
    that have stalled, and retries the n8n webhook.
    """
    with app.app_context():
        while True:
            try:
                # Look for entries pending for more than 2 minutes and retry up to 5 times
                two_mins_ago = datetime.utcnow() - timedelta(minutes=2)
                pending_entries = CatalogEntry.query.filter(
                    CatalogEntry.status == 'pending',
                    CatalogEntry.updated_at < two_mins_ago,
                    CatalogEntry.retry_count < 5
                ).all()

                if pending_entries:
                    print(f"Found {len(pending_entries)} stalled pending entries. Triggering batch retry...")

                for entry in pending_entries:
                    print(f"Retrying entry {entry.id} (Attempt {entry.retry_count + 1})...")
                    entry.retry_count += 1
                    
                    # We commit right away to update the updated_at timestamp and retry_count
                    db.session.commit()
                    
                    try:
                        requests.post(N8N_WEBHOOK_URL, json={
                            "entry_id": entry.id, 
                            "raw_input": entry.raw_input, 
                            "input_type": entry.input_type
                        }, timeout=5.0)
                    except requests.exceptions.RequestException as e:
                        print(f"Error triggering n8n webhook on retry: {e}")
                
                # Mark as failed if retry count exceeded
                failed_entries = CatalogEntry.query.filter(
                    CatalogEntry.status == 'pending',
                    CatalogEntry.retry_count >= 5
                ).all()
                for entry in failed_entries:
                    print(f"Max retries reached for {entry.id}. Marking as failed.")
                    entry.status = 'failed'
                    db.session.commit()
                    
            except Exception as e:
                print(f"Error in background retry task: {e}")
                
            time.sleep(60) # Run check every 60 seconds

@app.route('/api/ideas/submit', methods=['POST'])
def submit_idea():
    data = request.json
    raw_input = data.get('raw_input')
    input_type = data.get('input_type', 'idea')
    
    if not raw_input:
        return jsonify({"error": "raw_input is required"}), 400

    # Ensure a default user exists for demonstration
    user = User.query.first()
    if not user:
        user = User(email="demo@ideaincubator.local")
        db.session.add(user)
        db.session.commit()

    entry_id = str(uuid.uuid4())
    new_entry = CatalogEntry(
        id=entry_id,
        user_id=user.id,
        raw_input=raw_input,
        input_type=input_type,
        status='pending'
    )
    db.session.add(new_entry)
    db.session.commit()

    # Trigger n8n Webhook asynchronously (in reality, use a task queue like Celery, but here we do a simple non-blocking request or just blocking since it's a prototype)
    # To prevent blocking, we just fire and forget if possible, or wait with a short timeout.
    try:
        # We pass the entry_id so n8n can callback and update the record
        requests.post(N8N_WEBHOOK_URL, json={"entry_id": entry_id, "raw_input": raw_input, "input_type": input_type}, timeout=5.0)
    except requests.exceptions.ReadTimeout:
        pass # Expected since we don't wait for response
    except requests.exceptions.RequestException as e:
        print(f"Error triggering n8n webhook: {e}")

    return jsonify({"message": "Idea submitted successfully", "entry_id": entry_id, "status": "pending"}), 201

@app.route('/api/catalogs/', methods=['GET'])
def get_catalogs():
    # Ensure clean session
    db.session.rollback()
    try:
        entries = CatalogEntry.query.order_by(CatalogEntry.created_at.desc()).all()
        result = []
        for e in entries:
            result.append({
                "id": e.id,
                "user_id": e.user_id,
                "raw_input": e.raw_input,
                "input_type": e.input_type,
                "status": e.status,
                "summary": e.summary,
                "tech_stack": e.tech_stack,
                "pros_cons": e.pros_cons,
                "similar_tools": e.similar_tools,
                "creator": e.creator,
                "link": e.link,
                "installation": e.installation,
                "unique_features": e.unique_features,
                "market_trend": e.market_trend,
                "mermaid_syntax": e.mermaid_syntax,
                "image_url": e.image_url,
                "created_at": e.created_at.isoformat(),
            })
        return jsonify(result), 200
    except Exception as e:
        print(f"Error fetching catalogs: {e}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/webhooks/n8n-callback', methods=['POST'])
def n8n_callback():
    try:
        # 1. Catch the standard text strings from the form data
        entry_id = request.form.get('entry_id')
        summary = request.form.get('summary')
        mermaid_syntax = request.form.get('mermaid_syntax', '')
        
        if not entry_id:
            return jsonify({"status": "error", "message": "entry_id is required"}), 400

        # 2. Catch and parse the JSON-stringified arrays/objects
        def safe_json_loads(key):
            val = request.form.get(key)
            if not val:
                return [] if key != 'pros_cons' else {"pros": [], "cons": []}
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return [] if key != 'pros_cons' else {"pros": [], "cons": []}

        tech_stack = safe_json_loads('tech_stack')
        pros_cons = safe_json_loads('pros_cons')
        similar_tools = safe_json_loads('similar_tools')
        tags = safe_json_loads('tags')
        unique_features = safe_json_loads('unique_features')
        
        creator = request.form.get('creator')
        link = request.form.get('link')
        installation = request.form.get('installation')
        market_trend = request.form.get('market_trend')

        # 3. Extract the uploaded image file
        image_file = request.files.get('image_file')
        image_url = None

        # 4. Secure the filename and save to static/uploads
        if image_file and image_file.filename != '':
            uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            
            filename = secure_filename(image_file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(uploads_dir, unique_filename)
            image_file.save(file_path)
            
            # Construct the absolute URL for the frontend
            image_url = f"{BACKEND_BASE_URL}/static/uploads/{unique_filename}"

        # 5. Placeholder comment for database update logic
        # TODO: Add your SQLAlchemy (or standard SQL) database update logic using the entry_id here
        
        # Actual Implementation:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"status": "error", "message": "CatalogEntry not found"}), 404

        entry.status = "completed"
        entry.summary = summary
        entry.tech_stack = tech_stack
        entry.pros_cons = pros_cons
        entry.similar_tools = similar_tools
        
        entry.creator = creator
        entry.link = link
        entry.installation = installation
        entry.unique_features = unique_features
        entry.market_trend = market_trend
        
        # Handle mermaid syntax formatting
        if isinstance(mermaid_syntax, str):
            try:
                parsed_mermaid = json.loads(mermaid_syntax)
                if isinstance(parsed_mermaid, str):
                    mermaid_syntax = parsed_mermaid
            except json.JSONDecodeError:
                pass

            if mermaid_syntax.startswith('"') and mermaid_syntax.endswith('"'):
                mermaid_syntax = mermaid_syntax[1:-1]
                
            mermaid_syntax = mermaid_syntax.replace('\\n', '\n')
            mermaid_syntax = mermaid_syntax.replace('```mermaid\n', '').replace('```mermaid', '').replace('```\n', '').replace('```', '').strip()
            
        entry.mermaid_syntax = mermaid_syntax
        
        if image_url:
            entry.image_url = image_url
            
        db.session.commit()
        
        # 7. Auto-generate embedding for vector search
        try:
            # Create text representation of the catalog entry for embedding
            text_parts = [
                entry.raw_input,
                entry.summary,
                entry.creator,
            ]
            if entry.tech_stack:
                text_parts.extend(entry.tech_stack if isinstance(entry.tech_stack, list) else [])
            if entry.unique_features:
                text_parts.extend(entry.unique_features if isinstance(entry.unique_features, list) else [])
            
            text_for_embedding = " ".join([part.strip() for part in text_parts if part and str(part).strip()])
            
            if text_for_embedding and embedding_model and index:
                # Generate embedding
                embedding_vector = embedding_model.encode(text_for_embedding).tolist()
                
                # Upsert to Pinecone
                try:
                    index.upsert(
                        vectors=[{
                            "id": entry.id,
                            "values": embedding_vector,
                            "metadata": {
                                "raw_input": entry.raw_input or "",
                                "summary": (entry.summary or "")[:500]
                            }
                        }]
                    )
                    print(f"[n8n_callback] Successfully indexed entry {entry.id} to Pinecone")
                except Exception as e:
                    print(f"[n8n_callback] Error upserting to Pinecone: {e}")
        except Exception as e:
            print(f"[n8n_callback] Error generating embedding: {e}")

        # 8. Return a standard JSON response
        return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"Webhook processing error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/catalogs/<entry_id>/chat', methods=['GET'])
def get_chat_messages(entry_id):
    """Get chat messages for a specific catalog entry"""
    try:
        messages = ChatMessage.query.filter_by(catalog_entry_id=entry_id).order_by(ChatMessage.created_at.asc()).all()
        result = []
        for message in messages:
            result.append({
                "id": message.id,
                "message": message.message,
                "is_user": message.is_user,
                "created_at": message.created_at.isoformat(),
                "user_id": message.user_id
            })
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/catalogs/<entry_id>/chat', methods=['POST'])
def send_chat_message(entry_id):
    """Send a new chat message for a catalog entry (called by frontend and n8n callback)"""
    try:
        data = request.json
        message_text = data.get('message')
        is_user = data.get('is_user', True)
        user_id = data.get('user_id')
        
        if not message_text:
            return jsonify({"error": "message is required"}), 400
        
        # Get or create user - handle both frontend user and n8n AI assistant
        db_user = None
        if user_id:
            db_user = User.query.filter_by(id=user_id).first()
        
        if not db_user:
            # Use default user if not found
            db_user = User.query.first()
            if not db_user:
                db_user = User(email="demo@ideaincubator.local")
                db.session.add(db_user)
                db.session.commit()
        
        # Verify catalog entry exists
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Catalog entry not found"}), 404
        
        new_message = ChatMessage(
            id=str(uuid.uuid4()),
            catalog_entry_id=entry_id,
            user_id=db_user.id,
            message=message_text,
            is_user=is_user
        )
        db.session.add(new_message)
        db.session.commit()
        
        return jsonify({
            "id": new_message.id,
            "message": new_message.message,
            "is_user": new_message.is_user,
            "created_at": new_message.created_at.isoformat(),
            "user_id": new_message.user_id
        }), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/catalogs/<entry_id>/generate-embedding', methods=['POST'])
def generate_catalog_embedding(entry_id):
    """Generate and store vector embedding for a catalog entry"""
    try:
        # Get the catalog entry
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Catalog entry not found"}), 404
        
        # Create text representation of the catalog entry for embedding
        text_parts = [
            entry.raw_input or "",
            entry.summary or "",
            " ".join(entry.tech_stack) if isinstance(entry.tech_stack, list) else str(entry.tech_stack or ""),
            " ".join(entry.pros_cons.get("pros", [])) if isinstance(entry.pros_cons, dict) else "",
            " ".join(entry.pros_cons.get("cons", [])) if isinstance(entry.pros_cons, dict) else "",
            " ".join(entry.similar_tools) if isinstance(entry.similar_tools, list) else str(entry.similar_tools or ""),
            entry.creator or "",
            entry.link or "",
            entry.installation or "",
            " ".join(entry.unique_features) if isinstance(entry.unique_features, list) else str(entry.unique_features or ""),
            entry.market_trend or ""
        ]
        
        # Join and clean text
        text_for_embedding = " ".join([part.strip() for part in text_parts if part and str(part).strip()])
        
        if not text_for_embedding:
            return jsonify({"error": "No content available for embedding"}), 400
        
        if not ENABLE_VECTOR_SEARCH or not embedding_model:
            return jsonify({"error": "Embedding feature is disabled on this server"}), 503

        # Generate embedding using SentenceTransformer
        try:
            embedding_vector = embedding_model.encode(text_for_embedding).tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return jsonify({"error": "Failed to generate embedding"}), 500
        
        # Store in Pinecone when configured.
        if index:
            try:
                index.upsert(
                    vectors=[{
                        "id": entry_id,
                        "values": embedding_vector,
                        "metadata": {
                            "raw_input": entry.raw_input or "",
                            "summary": entry.summary or "",
                            "input_type": entry.input_type
                        }
                    }]
                )
            except Exception as e:
                print(f"Error upserting to Pinecone: {e}")
        
        # Check if embedding already exists in PostgreSQL
        existing_embedding = CatalogEmbedding.query.filter_by(catalog_entry_id=entry_id).first()
        if existing_embedding:
            # Update existing embedding
            existing_embedding.embedding = embedding_vector
            existing_embedding.embedding_json = {"vector": embedding_vector}
            existing_embedding.updated_at = datetime.utcnow()
        else:
            # Create new embedding
            new_embedding = CatalogEmbedding(
                id=str(uuid.uuid4()),
                catalog_entry_id=entry_id,
                embedding=embedding_vector,
                embedding_json={"vector": embedding_vector}
            )
            db.session.add(new_embedding)
        
        db.session.commit()
        
        return jsonify({
            "message": "Embedding generated successfully",
            "entry_id": entry_id,
            "embedding_dimension": len(embedding_vector)
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/catalogs/search', methods=['POST'])
def search_catalogs():
    """Search catalog entries using vector similarity with Pinecone"""
    try:
        data = request.json
        query_text = data.get('query')
        limit = data.get('limit', 10)
        
        if not query_text:
            return jsonify({"error": "query is required"}), 400
        
        if not ENABLE_VECTOR_SEARCH or not embedding_model:
            entries = CatalogEntry.query.filter(
                db.or_(
                    CatalogEntry.raw_input.ilike(f'%{query_text}%'),
                    CatalogEntry.summary.ilike(f'%{query_text}%'),
                    CatalogEntry.tech_stack.cast(db.String).ilike(f'%{query_text}%'),
                    CatalogEntry.creator.ilike(f'%{query_text}%')
                )
            ).limit(limit).all()

            result = []
            for entry in entries:
                result.append({
                    "id": entry.id,
                    "raw_input": entry.raw_input,
                    "input_type": entry.input_type,
                    "status": entry.status,
                    "summary": entry.summary,
                    "tech_stack": entry.tech_stack,
                    "pros_cons": entry.pros_cons,
                    "similar_tools": entry.similar_tools,
                    "creator": entry.creator,
                    "link": entry.link,
                    "installation": entry.installation,
                    "unique_features": entry.unique_features,
                    "market_trend": entry.market_trend,
                    "mermaid_syntax": entry.mermaid_syntax,
                    "image_url": entry.image_url,
                    "created_at": entry.created_at.isoformat(),
                })
            return jsonify(result), 200

        # Generate embedding for query text using SentenceTransformer
        try:
            query_embedding = embedding_model.encode(query_text).tolist()
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return jsonify({"error": "Failed to generate query embedding"}), 500
        
        # Search in Pinecone when configured.
        try:
            if not index:
                raise RuntimeError("Pinecone index not configured")

            search_results = index.query(
                vector=query_embedding,
                top_k=limit,
                include_metadata=True
            )
            
            # Get entry IDs from results and fetch full entries from PostgreSQL
            entry_ids = [match['id'] for match in search_results['matches']]
            
            # If Pinecone returns results, use them; otherwise fall back to text search
            if entry_ids:
                entries = CatalogEntry.query.filter(CatalogEntry.id.in_(entry_ids)).all()
                # Order by search relevance (Pinecone returns results sorted by score)
                entry_dict = {entry.id: entry for entry in entries}
                ordered_entries = [entry_dict[eid] for eid in entry_ids if eid in entry_dict]
            else:
                # No Pinecone results - fall back to text search
                print("No Pinecone results, falling back to text search...")
                entries = CatalogEntry.query.filter(
                    db.or_(
                        CatalogEntry.raw_input.ilike(f'%{query_text}%'),
                        CatalogEntry.summary.ilike(f'%{query_text}%'),
                        CatalogEntry.tech_stack.cast(db.String).ilike(f'%{query_text}%'),
                        CatalogEntry.creator.ilike(f'%{query_text}%')
                    )
                ).limit(limit).all()
                ordered_entries = entries
                print(f"Text search returned {len(ordered_entries)} results")
            
            # Format response
            result = []
            for entry in ordered_entries:
                result.append({
                    "id": entry.id,
                    "raw_input": entry.raw_input,
                    "input_type": entry.input_type,
                    "status": entry.status,
                    "summary": entry.summary,
                    "tech_stack": entry.tech_stack,
                    "pros_cons": entry.pros_cons,
                    "similar_tools": entry.similar_tools,
                    "creator": entry.creator,
                    "link": entry.link,
                    "installation": entry.installation,
                    "unique_features": entry.unique_features,
                    "market_trend": entry.market_trend,
                    "mermaid_syntax": entry.mermaid_syntax,
                    "image_url": entry.image_url,
                    "created_at": entry.created_at.isoformat(),
                })
            
            return jsonify(result), 200
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            # Fallback to text-based search if needed
            entries = CatalogEntry.query.filter(
                db.or_(
                    CatalogEntry.raw_input.ilike(f'%{query_text}%'),
                    CatalogEntry.summary.ilike(f'%{query_text}%')
                )
            ).limit(limit).all()
            
            result = []
            for entry in entries:
                result.append({
                    "id": entry.id,
                    "raw_input": entry.raw_input,
                    "input_type": entry.input_type,
                    "status": entry.status,
                    "summary": entry.summary,
                    "tech_stack": entry.tech_stack,
                    "pros_cons": entry.pros_cons,
                    "similar_tools": entry.similar_tools,
                    "creator": entry.creator,
                    "link": entry.link,
                    "installation": entry.installation,
                    "unique_features": entry.unique_features,
                    "market_trend": entry.market_trend,
                    "mermaid_syntax": entry.mermaid_syntax,
                    "image_url": entry.image_url,
                    "created_at": entry.created_at.isoformat(),
                })
            
            return jsonify(result), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/webhooks/chat-message', methods=['POST'])
def chat_message_webhook():
    """Endpoint for frontend to initiate chat, which forwards to n8n"""
    # Ensure clean session state
    db.session.rollback()
    
    try:
        # Get JSON data
        try:
            data = request.get_json(silent=True)
        except Exception as e:
            print(f"[chat-message webhook] Error parsing JSON: {e}")
            data = None
        
        print(f"[chat-message webhook] Received data: {data}, type: {type(data)}")
        
        if data is None:
            return jsonify({"error": "Invalid JSON or no data provided", "received": str(request.data)}), 400
            
        entry_id = data.get('entry_id') if data else None
        user_message = data.get('message') if data else None
        user_id = data.get('user_id') if data else None
        
        print(f"[chat-message webhook] entry_id: {entry_id}, user_message: {user_message}, user_id: {user_id}")
        
        if not entry_id or not user_message:
            return jsonify({"error": "entry_id and message are required", "received": data}), 400
        
        # Get or create user - ensure we always have a valid user_id from the users table
        db_user = None
        
        # First try to find the user by provided id
        if user_id:
            db_user = User.query.filter_by(id=user_id).first()
        
        # If not found or no user_id provided, get or create default user
        if not db_user:
            db_user = User.query.first()
            if not db_user:
                # Create default user
                db_user = User(email="demo@ideaincubator.local")
                db.session.add(db_user)
                db.session.commit()
                print(f"[chat-message webhook] Created default user: {db_user.id}")
        
        print(f"[chat-message webhook] Using user_id: {db_user.id}")
        
        # Store user message with valid user_id
        try:
            user_chat = ChatMessage(
                id=str(uuid.uuid4()),
                catalog_entry_id=entry_id,
                user_id=db_user.id,
                message=user_message,
                is_user=True
            )
            db.session.add(user_chat)
            db.session.commit()
            print(f"[chat-message webhook] User message stored successfully with user_id: {db_user.id}")
        except Exception as e:
            print(f"[chat-message webhook] Error storing user message: {e}")
            db.session.rollback()
            return jsonify({"status": "error", "message": f"Error storing message: {str(e)}"}), 400
        
        # Get catalog context for n8n
        try:
            entry = CatalogEntry.query.get(entry_id)
            if not entry:
                return jsonify({"error": "Catalog entry not found"}), 404
                
            print(f"[chat-message webhook] Found entry: {entry.id}, summary: {entry.summary}")
            
            # Forward to n8n webhook
            n8n_payload = {
                "entry_id": entry.id,
                "message": user_message,
                "user_id": db_user.id,
                "catalog_context": {
                    "summary": entry.summary or "",
                    "tech_stack": entry.tech_stack or [],
                    "pros_cons": entry.pros_cons or {"pros": [], "cons": []},
                    "similar_tools": entry.similar_tools or [],
                    "raw_input": entry.raw_input or ""
                }
            }
            
            try:
                n8n_response = requests.post(
                    N8N_CHAT_WEBHOOK_URL,
                    json=n8n_payload,
                    timeout=10.0
                )
                print(f"[chat-message webhook] n8n response status: {n8n_response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"[chat-message webhook] Error forwarding to n8n: {e}")
                # Continue - n8n might be unreachable but we already saved user message
            
            return jsonify({
                "status": "processing",
                "entry_id": entry.id,
                "message": "Message received and processing started"
            }), 200
            
        except Exception as e:
            print(f"[chat-message webhook] Error getting catalog entry: {e}")
            return jsonify({"status": "error", "message": f"Error getting entry: {str(e)}"}), 400
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Start background thread
retry_thread = threading.Thread(target=background_retry_job, args=(app,), daemon=True)
retry_thread.start()

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False) # disabled reloader to prevent duplicate threads
