import sys
import os
import base64
from functools import wraps
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from models import db, User, CatalogEntry, ChatMessage, CatalogEmbedding, Comment, Like, Bookmark, Reaction, ConnectRequest, Notification, IdeaUpdate, Collaborator, REACTION_TYPES, CONNECT_ROLES
from datetime import datetime
import uuid
import requests as http_requests
import json
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
import threading
import time
from datetime import timedelta

Pinecone = None
ServerlessSpec = None
load_dotenv()

# Vector services are optional. Keep disabled on small instances by default.
embedding_model = None
pc = None
index = None
ENABLE_VECTOR_SEARCH = os.getenv('ENABLE_VECTOR_SEARCH', 'false').lower() == 'true'

if ENABLE_VECTOR_SEARCH:
    try:
        from sentence_transformers import SentenceTransformer
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

def normalize_database_url(database_url):
    if database_url.startswith('postgres://'):
        return database_url.replace('postgres://', 'postgresql://', 1)
    return database_url

cors_origins_raw = os.getenv('CORS_ALLOWED_ORIGINS', '').strip()
if cors_origins_raw:
    cors_origins = [origin.strip() for origin in cors_origins_raw.split(',') if origin.strip()]
    CORS(app, resources={r"/api/*": {"origins": cors_origins, "allow_headers": ["Authorization", "Content-Type"]}})
else:
    CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Authorization", "Content-Type"]}})

# Respect X-Forwarded-* headers from Render/Netlify proxies.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Database Configuration
db_uri = normalize_database_url(os.getenv('DATABASE_URL', 'sqlite:///incubator.db'))
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

engine_options = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

if db_uri.startswith('postgresql'):
    print("Detected PostgreSQL - adding SSL and pool options", file=sys.stderr)
    engine_options['pool_size'] = 10
    engine_options['max_overflow'] = 20
    engine_options['connect_args'] = {
        'connect_timeout': 15,
        'sslmode': 'require'
    }

    # Force IPv4 to prevent connection fallback delays or issues with IPv6
    import socket
    from urllib.parse import urlparse
    try:
        parsed = urlparse(db_uri)
        if parsed.hostname and not parsed.hostname.replace('.', '').isdigit():
            ipv4 = socket.gethostbyname(parsed.hostname)
            engine_options['connect_args']['host'] = ipv4
            print(f"Resolved DB host {parsed.hostname} to {ipv4} for stability", file=sys.stderr)
    except Exception as e:
        print(f"Failed to resolve DB host: {e}", file=sys.stderr)

app.config['SQLALCHEMY_ENGINE_OPTIONS'] = engine_options

db.init_app(app)

def init_db():
    """Create tables, retrying up to 5 times with back-off to survive slow cold starts."""
    import time as _time
    for attempt in range(1, 6):
        try:
            db.create_all()
            test_db_connection()
            print("Database initialized OK", file=sys.stderr)
            return
        except Exception as e:
            print(f"DB init attempt {attempt}/5 failed: {e}", file=sys.stderr)
            _time.sleep(2)


# Test database connection
def test_db_connection():
    try:
        db.session.execute(db.text('SELECT 1'))
        print("Database connection OK", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Database connection error: {e}", file=sys.stderr)
        return False

N8N_WEBHOOK_URL = os.getenv('N8N_WEBHOOK_URL', 'http://localhost:5678/webhook/idea-incubator')
N8N_CHAT_WEBHOOK_URL = os.getenv('N8N_CHAT_WEBHOOK_URL', 'http://localhost:5678/webhook/chat-message')
BACKEND_BASE_URL = os.getenv('BACKEND_BASE_URL', '').rstrip('/')

# Supabase Auth config
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://vwuwrvxlcykurihjagcp.supabase.co')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY', '')

# Simple in-memory cache for auth tokens (token -> {user, expires_at})
_auth_cache = {}
_AUTH_CACHE_TTL = 300  # 5 minutes


def verify_supabase_token(token):
    """Verify a Supabase JWT by calling the Supabase Auth API. Results are cached."""
    now = time.time()

    # Check cache first
    cached = _auth_cache.get(token)
    if cached and cached['expires_at'] > now:
        return cached['user']

    try:
        resp = http_requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_ANON_KEY
            },
            timeout=5.0
        )
        if resp.status_code == 200:
            user_data = resp.json()
            _auth_cache[token] = {'user': user_data, 'expires_at': now + _AUTH_CACHE_TTL}
            return user_data
    except Exception as e:
        print(f"Supabase auth verification error: {e}")
    return None


def auth_required(f):
    """Decorator that requires a valid Supabase JWT in the Authorization header."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header required"}), 401

        token = auth_header[7:]
        supabase_user = verify_supabase_token(token)
        if not supabase_user:
            return jsonify({"error": "Invalid or expired token"}), 401

        # Sync user to our DB (create if not exists)
        supabase_id = supabase_user.get('id')
        email = supabase_user.get('email', '')
        name = supabase_user.get('user_metadata', {}).get('name', '')

        db_user = User.query.filter_by(id=supabase_id).first()
        if not db_user:
            # Check if this is the first real user — migrate demo entries
            demo_user = User.query.filter_by(email='demo@ideaincubator.local').first()
            db_user = User(id=supabase_id, email=email, name=name)
            db.session.add(db_user)
            db.session.commit()

            if demo_user:
                # Migrate all demo entries to this first real user
                CatalogEntry.query.filter_by(user_id=demo_user.id).update({'user_id': supabase_id})
                ChatMessage.query.filter_by(user_id=demo_user.id).update({'user_id': supabase_id})
                db.session.delete(demo_user)
                db.session.commit()
                print(f"Migrated demo entries to new user {email} ({supabase_id})")
        else:
            # Update name/email if changed
            if email and db_user.email != email:
                db_user.email = email
            if name and db_user.name != name:
                db_user.name = name
            db.session.commit()

        g.user_id = supabase_id
        g.user_email = email
        g.user_name = name
        return f(*args, **kwargs)
    return decorated


def get_public_backend_base_url():
    if BACKEND_BASE_URL:
        return BACKEND_BASE_URL
    return request.host_url.rstrip('/')

@app.route('/')
def root_status():
    return jsonify({
        "service": "MindInspo Backend",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "docs_hint": "Use /api/catalogs/ or /api/community/feed"
    }), 200

@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"}), 200

@app.route('/api/stats', methods=['GET'])
def public_stats():
    try:
        total_users = User.query.count()
        total_ideas = CatalogEntry.query.filter_by(status='completed').count()
        public_ideas = CatalogEntry.query.filter_by(visibility='public', status='completed').count()
        recent_public = CatalogEntry.query.filter_by(
            visibility='public', status='completed'
        ).order_by(CatalogEntry.published_at.desc()).limit(3).all()

        def mini_serialize(e):
            tags = e.tags or []
            return {
                'id': e.id,
                'raw_input': e.raw_input,
                'summary': (e.summary or '')[:120],
                'tags': tags[:3],
                'image_url': e.image_url,
            }

        recent_users = User.query.order_by(User.created_at.desc()).limit(4).all()

        def user_preview(u):
            display = u.name or u.email or ''
            initials = ''.join(p[0].upper() for p in display.split()[:2]) if display else '?'
            return {'initials': initials, 'avatar_url': u.avatar_url}

        return jsonify({
            'total_users': total_users,
            'total_ideas': total_ideas,
            'public_ideas': public_ideas,
            'recent': [mini_serialize(e) for e in recent_public],
            'recent_users': [user_preview(u) for u in recent_users],
        })
    except Exception:
        return jsonify({'total_users': 0, 'total_ideas': 0, 'public_ideas': 0, 'recent': [], 'recent_users': []}), 200


@app.route('/api/account/delete', methods=['DELETE'])
@auth_required
def delete_account():
    user_id = g.user_id
    try:
        Notification.query.filter_by(user_id=user_id).delete()
        ConnectRequest.query.filter(
            (ConnectRequest.requester_id == user_id) | (ConnectRequest.owner_id == user_id)
        ).delete(synchronize_session=False)
        Collaborator.query.filter_by(user_id=user_id).delete()
        IdeaUpdate.query.filter(
            IdeaUpdate.catalog_entry_id.in_(
                db.session.query(CatalogEntry.id).filter_by(user_id=user_id)
            )
        ).delete(synchronize_session=False)
        Reaction.query.filter_by(user_id=user_id).delete()
        Bookmark.query.filter_by(user_id=user_id).delete()
        Comment.query.filter_by(user_id=user_id).delete()
        ChatMessage.query.filter_by(user_id=user_id).delete()
        CatalogEntry.query.filter_by(user_id=user_id).delete()
        User.query.filter_by(id=user_id).delete()
        db.session.commit()
        return jsonify({'message': 'Account deleted'}), 200
    except Exception as e:
        db.session.rollback()
        print(f"Account delete error for {user_id}: {e}")
        return jsonify({'error': 'Failed to delete account'}), 500


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
                        http_requests.post(N8N_WEBHOOK_URL, json={
                            "entry_id": entry.id, 
                            "raw_input": entry.raw_input, 
                            "input_type": entry.input_type
                        }, timeout=5.0)
                    except http_requests.exceptions.RequestException as e:
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
@auth_required
def submit_idea():
    data = request.json
    raw_input = data.get('raw_input')
    input_type = data.get('input_type', 'idea')
    
    if not raw_input:
        return jsonify({"error": "raw_input is required"}), 400

    entry_id = str(uuid.uuid4())
    new_entry = CatalogEntry(
        id=entry_id,
        user_id=g.user_id,
        raw_input=raw_input,
        input_type=input_type,
        status='pending'
    )
    db.session.add(new_entry)
    db.session.commit()

    try:
        http_requests.post(N8N_WEBHOOK_URL, json={"entry_id": entry_id, "raw_input": raw_input, "input_type": input_type}, timeout=5.0)
    except http_requests.exceptions.ReadTimeout:
        pass
    except http_requests.exceptions.RequestException as e:
        print(f"Error triggering n8n webhook: {e}")

    return jsonify({"message": "Idea submitted successfully", "entry_id": entry_id, "status": "pending"}), 201

@app.route('/api/catalogs/', methods=['GET'])
@auth_required
def get_catalogs():
    # Ensure clean session
    db.session.rollback()
    try:
        # Get entries owned by user
        owned_entries = CatalogEntry.query.filter_by(user_id=g.user_id).all()
        
        # Get entries where user is a collaborator
        collab_entries = CatalogEntry.query.join(Collaborator).filter(Collaborator.user_id == g.user_id).all()
        
        # Combine, remove duplicates (if any), and sort by created_at desc
        entries = list(set(owned_entries + collab_entries))
        entries.sort(key=lambda x: x.created_at if x.created_at else datetime.min, reverse=True)
        
        result = serialize_entries_batch(entries, g.user_id)
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

        # 4. Upload to ImgBB (or fallback to local if it fails)
        if image_file and image_file.filename != '':
            filename = secure_filename(image_file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            
            IMGBB_API_KEY = os.getenv('IMGBB_API_KEY', 'cc532d52d3ec271f34a5dd8227db219a')
            
            try:
                image_content = image_file.read()
                b64_image = base64.b64encode(image_content).decode('utf-8')
                
                imgbb_res = http_requests.post(
                    "https://api.imgbb.com/1/upload",
                    data={"key": IMGBB_API_KEY, "image": b64_image, "name": filename},
                    timeout=20.0
                )
                
                if imgbb_res.status_code == 200:
                    image_url = imgbb_res.json()['data']['url']
                    print(f"Image uploaded to ImgBB successfully: {image_url}")
                else:
                    raise Exception(f"ImgBB returned {imgbb_res.status_code}: {imgbb_res.text}")
            except Exception as e:
                print(f"ImgBB upload failed, falling back to local storage: {e}")
                uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
                os.makedirs(uploads_dir, exist_ok=True)
                file_path = os.path.join(uploads_dir, unique_filename)
                image_file.seek(0)
                image_file.save(file_path)
                image_url = f"{get_public_backend_base_url()}/static/uploads/{unique_filename}"

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
@auth_required
def get_chat_messages(entry_id):
    """Get chat messages for a specific catalog entry"""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Catalog entry not found"}), 404
        # Check if owner or collaborator
        is_collab = Collaborator.query.filter_by(user_id=g.user_id, catalog_entry_id=entry_id).first()
        if entry.user_id != g.user_id and not is_collab:
            return jsonify({"error": "Unauthorized"}), 403

        messages = ChatMessage.query.filter_by(catalog_entry_id=entry_id).order_by(ChatMessage.created_at.asc()).all()
        result = []
        for message in messages:
            msg_data = {
                "id": message.id,
                "message": message.message,
                "is_user": message.is_user,
                "created_at": message.created_at.isoformat(),
                "user_id": message.user_id
            }
            # Include proposed_changes if stored in the message JSON
            try:
                parsed = json.loads(message.message)
                if isinstance(parsed, dict) and 'proposed_changes' in parsed:
                    msg_data['message'] = parsed.get('text', message.message)
                    msg_data['proposed_changes'] = parsed['proposed_changes']
            except (json.JSONDecodeError, TypeError):
                pass
            result.append(msg_data)
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
@auth_required
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
                CatalogEntry.user_id == g.user_id,
                db.or_(
                    CatalogEntry.raw_input.ilike(f'%{query_text}%'),
                    CatalogEntry.summary.ilike(f'%{query_text}%'),
                    CatalogEntry.tech_stack.cast(db.String).ilike(f'%{query_text}%'),
                    CatalogEntry.creator.ilike(f'%{query_text}%')
                )
            ).limit(limit).all()

            result = serialize_entries_batch(entries, g.user_id)
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
                entries = CatalogEntry.query.filter(
                    CatalogEntry.id.in_(entry_ids),
                    CatalogEntry.user_id == g.user_id
                ).all()
                # Order by search relevance (Pinecone returns results sorted by score)
                entry_dict = {entry.id: entry for entry in entries}
                ordered_entries = [entry_dict[eid] for eid in entry_ids if eid in entry_dict]
            else:
                # No Pinecone results - fall back to text search
                print("No Pinecone results, falling back to text search...")
                entries = CatalogEntry.query.filter(
                    CatalogEntry.user_id == g.user_id,
                    db.or_(
                        CatalogEntry.raw_input.ilike(f'%{query_text}%'),
                        CatalogEntry.summary.ilike(f'%{query_text}%'),
                        CatalogEntry.tech_stack.cast(db.String).ilike(f'%{query_text}%'),
                        CatalogEntry.creator.ilike(f'%{query_text}%')
                    )
                ).limit(limit).all()
                ordered_entries = entries
                print(f"Text search returned {len(ordered_entries)} results")
            
            result = serialize_entries_batch(ordered_entries, g.user_id)
            return jsonify(result), 200
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            # Fallback to text-based search if needed
            entries = CatalogEntry.query.filter(
                CatalogEntry.user_id == g.user_id,
                db.or_(
                    CatalogEntry.raw_input.ilike(f'%{query_text}%'),
                    CatalogEntry.summary.ilike(f'%{query_text}%')
                )
            ).limit(limit).all()
            
            result = serialize_entries_batch(entries, g.user_id)
            return jsonify(result), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/webhooks/chat-message', methods=['POST'])
@auth_required
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
        user_id = g.user_id
        
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
                n8n_response = http_requests.post(
                    N8N_CHAT_WEBHOOK_URL,
                    json=n8n_payload,
                    timeout=10.0
                )
                print(f"[chat-message webhook] n8n response status: {n8n_response.status_code}")
            except http_requests.exceptions.RequestException as e:
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

@app.route('/api/catalogs/<entry_id>/apply-chat-edit', methods=['POST'])
@auth_required
def apply_chat_edit(entry_id):
    """Apply proposed changes from chat to a catalog entry."""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Catalog entry not found"}), 404
        if entry.user_id != g.user_id:
            return jsonify({"error": "Unauthorized"}), 403

        data = request.json
        changes = data.get('changes', {})
        
        EDITABLE_FIELDS = ['summary', 'tech_stack', 'pros_cons', 'similar_tools',
                           'creator', 'link', 'installation', 'unique_features',
                           'market_trend', 'mermaid_syntax']
        
        applied = []
        for field, value in changes.items():
            if field in EDITABLE_FIELDS:
                setattr(entry, field, value)
                applied.append(field)

        if applied:
            entry.updated_at = datetime.utcnow()
            db.session.commit()

        return jsonify({
            "status": "success",
            "applied_fields": applied,
            "entry": serialize_entry(entry, g.user_id)
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/catalogs/<entry_id>', methods=['PUT'])
@auth_required
def update_catalog_entry(entry_id):
    """Owner-only manual edit endpoint for a catalog entry."""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Catalog entry not found"}), 404
        # Check if owner or collaborator
        is_collab = Collaborator.query.filter_by(user_id=g.user_id, catalog_entry_id=entry_id).first()
        if entry.user_id != g.user_id and not is_collab:
            return jsonify({"error": "Unauthorized"}), 403

        payload = request.json or {}
        editable_fields = {
            'raw_input': str,
            'summary': str,
            'tech_stack': (list, str),
            'pros_cons': (dict, str),
            'similar_tools': (list, str),
            'creator': str,
            'link': str,
            'installation': str,
            'unique_features': (list, str),
            'market_trend': str,
            'mermaid_syntax': str,
            'tags': (list, str),
        }

        updated_fields = []
        for field, expected_types in editable_fields.items():
            if field in payload:
                value = payload[field]
                if value is not None and not isinstance(value, expected_types):
                    return jsonify({"error": f"Invalid type for '{field}'"}), 400
                setattr(entry, field, value)
                updated_fields.append(field)

        if not updated_fields:
            return jsonify({"error": "No editable fields provided"}), 400

        entry.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            "status": "updated",
            "updated_fields": updated_fields,
            "entry": serialize_entry(entry, g.user_id)
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/webhooks/chat-response', methods=['POST'])
def chat_response_webhook():
    """Callback from n8n with AI chat response. May include proposed_changes."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON data"}), 400
        
        entry_id = data.get('entry_id')
        ai_message = data.get('message', '')
        proposed_changes = data.get('proposed_changes')  # optional dict
        
        if not entry_id or not ai_message:
            return jsonify({"error": "entry_id and message are required"}), 400
        
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Catalog entry not found"}), 404
        
        # Get the entry owner as the user for this message
        owner_id = entry.user_id
        
        # If there are proposed changes, store them inside the message JSON
        if proposed_changes and isinstance(proposed_changes, dict):
            message_content = json.dumps({
                "text": ai_message,
                "proposed_changes": proposed_changes
            })
        else:
            message_content = ai_message
        
        ai_chat = ChatMessage(
            id=str(uuid.uuid4()),
            catalog_entry_id=entry_id,
            user_id=owner_id,
            message=message_content,
            is_user=False
        )
        db.session.add(ai_chat)
        db.session.commit()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Chat response webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


# ============================================================
# Community API Endpoints
# ============================================================

def compute_idea_score(entry_id):
    """Calculate an idea score based on weighted reactions + comments + bookmarks."""
    weights = {'brilliant': 3, 'interested': 1, 'sellable': 4, 'build_worthy': 3, 'needs_work': 0.5}
    score = 0.0
    for rtype, w in weights.items():
        score += Reaction.query.filter_by(catalog_entry_id=entry_id, reaction_type=rtype).count() * w
    score += Comment.query.filter_by(catalog_entry_id=entry_id).count() * 2
    score += Bookmark.query.filter_by(catalog_entry_id=entry_id).count() * 1.5
    return round(score, 1)


def serialize_entries_batch(entries, user_id=None):
    """Batch serialize multiple entries with minimal DB queries (N+1 fix)."""
    if not entries:
        return []
    entry_ids = [e.id for e in entries]
    owner_ids = list(set(e.user_id for e in entries))

    from concurrent.futures import ThreadPoolExecutor
    from flask import current_app
    app_obj = current_app._get_current_object()

    def fetch_in_context(func):
        with app_obj.app_context():
            try:
                return func()
            finally:
                db.session.remove()

    with ThreadPoolExecutor(max_workers=10) as executor:
        f_likes = executor.submit(fetch_in_context, lambda: dict(db.session.query(Like.catalog_entry_id, db.func.count(Like.id)).filter(Like.catalog_entry_id.in_(entry_ids)).group_by(Like.catalog_entry_id).all()))
        f_comments = executor.submit(fetch_in_context, lambda: dict(db.session.query(Comment.catalog_entry_id, db.func.count(Comment.id)).filter(Comment.catalog_entry_id.in_(entry_ids)).group_by(Comment.catalog_entry_id).all()))
        f_bookmarks = executor.submit(fetch_in_context, lambda: dict(db.session.query(Bookmark.catalog_entry_id, db.func.count(Bookmark.id)).filter(Bookmark.catalog_entry_id.in_(entry_ids)).group_by(Bookmark.catalog_entry_id).all()))
        f_connects = executor.submit(fetch_in_context, lambda: dict(db.session.query(ConnectRequest.catalog_entry_id, db.func.count(ConnectRequest.id)).filter(ConnectRequest.catalog_entry_id.in_(entry_ids), ConnectRequest.status == 'accepted').group_by(ConnectRequest.catalog_entry_id).all()))
        f_updates = executor.submit(fetch_in_context, lambda: dict(db.session.query(IdeaUpdate.catalog_entry_id, db.func.count(IdeaUpdate.id)).filter(IdeaUpdate.catalog_entry_id.in_(entry_ids)).group_by(IdeaUpdate.catalog_entry_id).all()))
        f_reactions = executor.submit(fetch_in_context, lambda: db.session.query(Reaction.catalog_entry_id, Reaction.reaction_type, db.func.count(Reaction.id)).filter(Reaction.catalog_entry_id.in_(entry_ids)).group_by(Reaction.catalog_entry_id, Reaction.reaction_type).all())
        f_authors = executor.submit(fetch_in_context, lambda: {u.id: u for u in User.query.filter(User.id.in_(owner_ids)).all()})
        f_collabs = executor.submit(fetch_in_context, lambda: db.session.query(Collaborator.catalog_entry_id, User, Collaborator.role).join(User, Collaborator.user_id == User.id).filter(Collaborator.catalog_entry_id.in_(entry_ids)).all())

        if user_id:
            f_user_likes = executor.submit(fetch_in_context, lambda: set(r[0] for r in db.session.query(Like.catalog_entry_id).filter(Like.catalog_entry_id.in_(entry_ids), Like.user_id == user_id).all()))
            f_user_bookmarks = executor.submit(fetch_in_context, lambda: set(r[0] for r in db.session.query(Bookmark.catalog_entry_id).filter(Bookmark.catalog_entry_id.in_(entry_ids), Bookmark.user_id == user_id).all()))
            f_user_reacts = executor.submit(fetch_in_context, lambda: db.session.query(Reaction.catalog_entry_id, Reaction.reaction_type).filter(Reaction.catalog_entry_id.in_(entry_ids), Reaction.user_id == user_id).all())
            f_user_connects = executor.submit(fetch_in_context, lambda: set(r[0] for r in db.session.query(ConnectRequest.catalog_entry_id).filter(ConnectRequest.catalog_entry_id.in_(entry_ids), ConnectRequest.requester_id == user_id).all()))

        # Await all futures
        like_counts = f_likes.result()
        comment_counts = f_comments.result()
        bookmark_counts = f_bookmarks.result()
        connect_counts = f_connects.result()
        update_counts = f_updates.result()
        reaction_rows = f_reactions.result()
        authors = f_authors.result()
        collab_rows = f_collabs.result()

        user_likes = set()
        user_bookmarks = set()
        user_reactions_map = {}
        user_connects = set()
        
        if user_id:
            user_likes = f_user_likes.result()
            user_bookmarks = f_user_bookmarks.result()
            for eid, rtype in f_user_reacts.result():
                user_reactions_map.setdefault(eid, []).append(rtype)
            user_connects = f_user_connects.result()

    reaction_map = {}
    for eid, rtype, cnt in reaction_rows:
        reaction_map.setdefault(eid, {})[rtype] = cnt

    collab_map = {}
    for eid, user_obj, role in collab_rows:
        collab_map.setdefault(eid, []).append({
            "id": user_obj.id,
            "name": user_obj.name or user_obj.email.split('@')[0],
            "avatar_url": user_obj.avatar_url,
            "role": role
        })

    results = []
    weights = {'brilliant': 3, 'interested': 1, 'sellable': 4, 'build_worthy': 3, 'needs_work': 0.5}
    for e in entries:
        reactions_summary = {rtype: reaction_map.get(e.id, {}).get(rtype, 0) for rtype in REACTION_TYPES}
        score = sum(reactions_summary.get(rt, 0) * w for rt, w in weights.items())
        score += comment_counts.get(e.id, 0) * 2
        score += bookmark_counts.get(e.id, 0) * 1.5

        data = {
            "id": e.id, "user_id": e.user_id, "raw_input": e.raw_input,
            "input_type": e.input_type, "status": e.status, "summary": e.summary,
            "tech_stack": e.tech_stack, "pros_cons": e.pros_cons,
            "similar_tools": e.similar_tools, "creator": e.creator, "link": e.link,
            "installation": e.installation, "unique_features": e.unique_features,
            "market_trend": e.market_trend, "mermaid_syntax": e.mermaid_syntax,
            "image_url": e.image_url,
            "visibility": getattr(e, 'visibility', 'private'),
            "published_at": e.published_at.isoformat() if getattr(e, 'published_at', None) else None,
            "view_count": getattr(e, 'view_count', 0),
            "tags": getattr(e, 'tags', None),
            "visible_fields": getattr(e, 'visible_fields', None),
            "created_at": e.created_at.isoformat(),
            "updated_at": e.updated_at.isoformat() if getattr(e, 'updated_at', None) else None,
            "like_count": like_counts.get(e.id, 0),
            "comment_count": comment_counts.get(e.id, 0),
            "bookmark_count": bookmark_counts.get(e.id, 0),
            "connect_count": connect_counts.get(e.id, 0),
            "updates_count": update_counts.get(e.id, 0),
            "reactions": reactions_summary,
            "idea_score": round(score, 1),
            "collaborators": collab_map.get(e.id, []),
        }
        author = authors.get(e.user_id)
        if author:
            data["author"] = {
                "id": author.id,
                "name": author.name or author.email.split('@')[0],
                "avatar_url": author.avatar_url,
            }
        if user_id:
            data["liked_by_user"] = e.id in user_likes
            data["bookmarked_by_user"] = e.id in user_bookmarks
            data["user_reactions"] = user_reactions_map.get(e.id, [])
            data["connect_sent"] = e.id in user_connects

        # Apply visible_fields filter — owner sees same as everyone else in community feed
        vis = getattr(e, 'visible_fields', None)
        if vis is not None:
            for field in ['summary', 'tech_stack', 'pros_cons', 'similar_tools',
                          'mermaid_syntax', 'image_url', 'market_trend', 'unique_features']:
                if field not in vis:
                    data[field] = None

        results.append(data)
    return results


def serialize_entry(e, user_id=None):
    """Serialize a CatalogEntry for API responses with community data."""
    data = {
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
        "visibility": getattr(e, 'visibility', 'private'),
        "published_at": e.published_at.isoformat() if getattr(e, 'published_at', None) else None,
        "view_count": getattr(e, 'view_count', 0),
        "tags": getattr(e, 'tags', None),
        "visible_fields": getattr(e, 'visible_fields', None),
        "created_at": e.created_at.isoformat(),
        "updated_at": e.updated_at.isoformat() if getattr(e, 'updated_at', None) else None,
    }
    # Add community stats
    data["like_count"] = Like.query.filter_by(catalog_entry_id=e.id).count()
    data["comment_count"] = Comment.query.filter_by(catalog_entry_id=e.id).count()
    data["bookmark_count"] = Bookmark.query.filter_by(catalog_entry_id=e.id).count()
    data["connect_count"] = ConnectRequest.query.filter_by(catalog_entry_id=e.id, status='accepted').count()
    data["updates_count"] = IdeaUpdate.query.filter_by(catalog_entry_id=e.id).count()

    # Reactions summary: {"brilliant": 5, "interested": 3, ...}
    reactions_summary = {}
    for rtype in REACTION_TYPES:
        reactions_summary[rtype] = Reaction.query.filter_by(catalog_entry_id=e.id, reaction_type=rtype).count()
    data["reactions"] = reactions_summary

    # Idea score
    data["idea_score"] = compute_idea_score(e.id)

    # Author info
    author = User.query.get(e.user_id)
    if author:
        data["author"] = {
            "id": author.id,
            "name": author.name or author.email.split('@')[0],
            "avatar_url": author.avatar_url,
        }
    # Current user interactions
    if user_id:
        data["liked_by_user"] = Like.query.filter_by(catalog_entry_id=e.id, user_id=user_id).first() is not None
        data["bookmarked_by_user"] = Bookmark.query.filter_by(catalog_entry_id=e.id, user_id=user_id).first() is not None
        # User's own reactions on this entry
        user_reactions = Reaction.query.filter_by(catalog_entry_id=e.id, user_id=user_id).all()
        data["user_reactions"] = [r.reaction_type for r in user_reactions]
        # Whether user has sent a connect request
        data["connect_sent"] = ConnectRequest.query.filter_by(catalog_entry_id=e.id, requester_id=user_id).first() is not None

    # Apply visible_fields filter for non-owner viewers
    vis = getattr(e, 'visible_fields', None)
    if vis and user_id != e.user_id:
        hideable = ['summary', 'tech_stack', 'pros_cons', 'similar_tools',
                    'mermaid_syntax', 'image_url', 'market_trend', 'unique_features']
        for field in hideable:
            if field not in vis:
                data[field] = None
    return data


import time
from cachetools import TTLCache
feed_cache = TTLCache(maxsize=100, ttl=15.0)

@app.route('/api/community/feed', methods=['GET'])
def community_feed():
    """Public community feed — returns published entries with 15s cache."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        sort = request.args.get('sort', 'recent')  # recent | popular
        filter_type = request.args.get('type', 'all')  # all | idea | tool
        search = request.args.get('q', '').strip()

        # Get user_id if authenticated
        user_id = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            su = verify_supabase_token(token)
            if su:
                user_id = su.get('id')

        # Cache key based on all parameters and user_id (for personalized likes/bookmarks)
        cache_key = f"feed_{page}_{per_page}_{sort}_{filter_type}_{search}_{user_id}"
        if cache_key in feed_cache:
            return jsonify({
                "entries": feed_cache[cache_key]["entries"],
                "total": feed_cache[cache_key]["total"],
                "pages": feed_cache[cache_key]["pages"],
                "current_page": page
            }), 200

        query = CatalogEntry.query.filter(
            CatalogEntry.visibility == 'public',
            CatalogEntry.status == 'completed'
        )

        if filter_type != 'all':
            query = query.filter(CatalogEntry.input_type == filter_type)

        if search:
            query = query.filter(
                db.or_(
                    CatalogEntry.raw_input.ilike(f'%{search}%'),
                    CatalogEntry.summary.ilike(f'%{search}%'),
                )
            )

        if sort == 'popular':
            query = query.order_by(CatalogEntry.view_count.desc(), CatalogEntry.published_at.desc())
        elif sort == 'trending':
            # Trending is handled post-query via idea_score
            query = query.order_by(CatalogEntry.published_at.desc())
        else:
            query = query.order_by(CatalogEntry.published_at.desc())
            
        t1 = time.time()

        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        t2 = time.time()
        t3 = time.time()

        result = serialize_entries_batch(list(pagination.items), user_id)
        
        # Sort by idea_score for trending
        if sort == 'trending':
            result.sort(key=lambda x: x.get('idea_score', 0), reverse=True)

        feed_cache[cache_key] = {
            "entries": result,
            "total": pagination.total,
            "pages": pagination.pages,
            "page": page,
            "per_page": per_page
        }

        return jsonify({
            "entries": result,
            "total": pagination.total,
            "page": page,
            "per_page": per_page,
            "pages": pagination.pages,
        }), 200
    except Exception as e:
        print(f"Error in community feed: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/api/community/leaderboard', methods=['GET'])
def community_leaderboard():
    """Returns top users based on community activity."""
    try:
        users = User.query.all()
        leaderboard = []
        
        for u in users:
            score = 0
            pub_count = CatalogEntry.query.filter_by(user_id=u.id, visibility='public').count()
            score += pub_count * 10
            
            update_count = IdeaUpdate.query.filter_by(user_id=u.id).count()
            score += update_count * 5
            
            entry_ids = [e.id for e in CatalogEntry.query.filter_by(user_id=u.id).all()]
            if entry_ids:
                likes_received = Like.query.filter(Like.catalog_entry_id.in_(entry_ids)).count()
                score += likes_received * 2
                
                reactions_received = Reaction.query.filter(Reaction.catalog_entry_id.in_(entry_ids)).count()
                score += reactions_received * 1
            
            if score > 0 or pub_count > 0:
                leaderboard.append({
                    "id": u.id,
                    "name": u.name or u.email.split('@')[0],
                    "avatar_url": u.avatar_url,
                    "score": score,
                    "published_count": pub_count,
                    "updates_count": update_count
                })
        
        leaderboard.sort(key=lambda x: x['score'], reverse=True)
        return jsonify(leaderboard[:10]), 200
    except Exception as e:
        print(f"Leaderboard error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/catalogs/<entry_id>/updates', methods=['GET'])
def list_idea_updates(entry_id):
    """Return timeline updates for a catalog entry (public or owner)."""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Entry not found"}), 404

        viewer_id = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            su = verify_supabase_token(token)
            if su:
                viewer_id = su.get('id')

        if entry.visibility != 'public' and entry.user_id != viewer_id:
            return jsonify({"error": "Unauthorized"}), 403

        updates = IdeaUpdate.query.filter_by(catalog_entry_id=entry_id).order_by(IdeaUpdate.created_at.desc()).all()
        return jsonify([
            {
                "id": u.id,
                "catalog_entry_id": u.catalog_entry_id,
                "user_id": u.user_id,
                "content": u.content,
                "update_type": u.update_type,
                "created_at": u.created_at.isoformat(),
                "updated_at": u.updated_at.isoformat() if u.updated_at else None,
            }
            for u in updates
        ]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/catalogs/<entry_id>/updates', methods=['POST'])
@auth_required
def create_idea_update(entry_id):
    """Create a new owner-only timeline update for a catalog entry."""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Entry not found"}), 404
        
        # Check if owner or collaborator
        is_collab = Collaborator.query.filter_by(user_id=g.user_id, catalog_entry_id=entry_id).first()
        if entry.user_id != g.user_id and not is_collab:
            return jsonify({"error": "Unauthorized"}), 403

        payload = request.json or {}
        content = (payload.get('content') or '').strip()
        update_type = (payload.get('update_type') or 'progress').strip().lower()
        allowed_types = {'progress', 'feedback', 'changelog', 'milestone'}

        if not content:
            return jsonify({"error": "content is required"}), 400
        if update_type not in allowed_types:
            return jsonify({"error": "Invalid update_type"}), 400

        update = IdeaUpdate(
            id=str(uuid.uuid4()),
            catalog_entry_id=entry_id,
            user_id=g.user_id,
            content=content,
            update_type=update_type,
        )
        db.session.add(update)
        db.session.commit()

        return jsonify({
            "status": "created",
            "update": {
                "id": update.id,
                "catalog_entry_id": update.catalog_entry_id,
                "user_id": update.user_id,
                "content": update.content,
                "update_type": update.update_type,
                "created_at": update.created_at.isoformat(),
                "updated_at": update.updated_at.isoformat() if update.updated_at else None,
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/catalogs/<entry_id>/updates/<update_id>', methods=['PUT'])
@auth_required
def update_idea_update(entry_id, update_id):
    """Edit an existing owner-authored timeline update."""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Entry not found"}), 404
        if entry.user_id != g.user_id:
            return jsonify({"error": "Unauthorized"}), 403

        update = IdeaUpdate.query.filter_by(id=update_id, catalog_entry_id=entry_id).first()
        if not update:
            return jsonify({"error": "Update not found"}), 404
        if update.user_id != g.user_id:
            return jsonify({"error": "Unauthorized"}), 403

        payload = request.json or {}
        if 'content' in payload:
            content = (payload.get('content') or '').strip()
            if not content:
                return jsonify({"error": "content cannot be empty"}), 400
            update.content = content

        if 'update_type' in payload:
            update_type = (payload.get('update_type') or '').strip().lower()
            if update_type not in {'progress', 'feedback', 'changelog', 'milestone'}:
                return jsonify({"error": "Invalid update_type"}), 400
            update.update_type = update_type

        update.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            "status": "updated",
            "update": {
                "id": update.id,
                "catalog_entry_id": update.catalog_entry_id,
                "user_id": update.user_id,
                "content": update.content,
                "update_type": update.update_type,
                "created_at": update.created_at.isoformat(),
                "updated_at": update.updated_at.isoformat() if update.updated_at else None,
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/catalogs/<entry_id>/publish', methods=['POST'])
@auth_required
def publish_entry(entry_id):
    """Publish a catalog entry to the community."""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Entry not found"}), 404
        if entry.user_id != g.user_id:
            return jsonify({"error": "Unauthorized"}), 403
        if entry.status != 'completed':
            return jsonify({"error": "Only completed entries can be published"}), 400

        data = request.json or {}
        entry.visibility = 'public'
        entry.published_at = datetime.utcnow()
        if data.get('tags'):
            entry.tags = data['tags']
        if 'visible_fields' in data:
            entry.visible_fields = data['visible_fields']  # list of field names or null
        db.session.commit()
        feed_cache.clear()

        return jsonify({"status": "published", "entry": serialize_entry(entry, g.user_id)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/catalogs/<entry_id>/unpublish', methods=['POST'])
@auth_required
def unpublish_entry(entry_id):
    """Unpublish a catalog entry from the community."""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Entry not found"}), 404
        if entry.user_id != g.user_id:
            return jsonify({"error": "Unauthorized"}), 403

        entry.visibility = 'private'
        entry.published_at = None
        db.session.commit()
        feed_cache.clear()

        return jsonify({"status": "unpublished"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Comments ----

@app.route('/api/community/<entry_id>/comments', methods=['GET'])
def get_comments(entry_id):
    """Get comments for a public entry."""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Entry not found"}), 404

        comments = Comment.query.filter_by(
            catalog_entry_id=entry_id, parent_id=None
        ).order_by(Comment.created_at.asc()).all()

        def serialize_comment(c):
            author = User.query.get(c.user_id)
            replies = Comment.query.filter_by(parent_id=c.id).order_by(Comment.created_at.asc()).all()
            return {
                "id": c.id,
                "content": c.content,
                "created_at": c.created_at.isoformat(),
                "author": {
                    "id": author.id,
                    "name": author.name or author.email.split('@')[0],
                    "avatar_url": author.avatar_url,
                } if author else None,
                "replies": [serialize_comment(r) for r in replies],
            }

        return jsonify([serialize_comment(c) for c in comments]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/community/<entry_id>/comments', methods=['POST'])
@auth_required
def post_comment(entry_id):
    """Post a comment on a public entry."""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Entry not found"}), 404

        data = request.json
        content = data.get('content', '').strip()
        parent_id = data.get('parent_id')  # for replies

        if not content:
            return jsonify({"error": "Content is required"}), 400

        comment = Comment(
            id=str(uuid.uuid4()),
            catalog_entry_id=entry_id,
            user_id=g.user_id,
            content=content,
            parent_id=parent_id,
        )
        db.session.add(comment)
        db.session.commit()

        author = User.query.get(g.user_id)
        return jsonify({
            "id": comment.id,
            "content": comment.content,
            "created_at": comment.created_at.isoformat(),
            "author": {
                "id": author.id,
                "name": author.name or author.email.split('@')[0],
                "avatar_url": author.avatar_url,
            } if author else None,
            "replies": [],
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Likes ----

@app.route('/api/community/<entry_id>/like', methods=['POST'])
@auth_required
def toggle_like(entry_id):
    """Toggle like on a public entry."""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Entry not found"}), 404

        existing = Like.query.filter_by(catalog_entry_id=entry_id, user_id=g.user_id).first()
        if existing:
            db.session.delete(existing)
            db.session.commit()
            return jsonify({"liked": False, "like_count": Like.query.filter_by(catalog_entry_id=entry_id).count()}), 200
        else:
            like = Like(id=str(uuid.uuid4()), catalog_entry_id=entry_id, user_id=g.user_id)
            db.session.add(like)
            db.session.commit()
            return jsonify({"liked": True, "like_count": Like.query.filter_by(catalog_entry_id=entry_id).count()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Bookmarks ----

@app.route('/api/community/<entry_id>/bookmark', methods=['POST'])
@auth_required
def toggle_bookmark(entry_id):
    """Toggle bookmark on an entry."""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Entry not found"}), 404

        existing = Bookmark.query.filter_by(catalog_entry_id=entry_id, user_id=g.user_id).first()
        if existing:
            db.session.delete(existing)
            db.session.commit()
            return jsonify({"bookmarked": False}), 200
        else:
            bm = Bookmark(id=str(uuid.uuid4()), catalog_entry_id=entry_id, user_id=g.user_id)
            db.session.add(bm)
            db.session.commit()
            return jsonify({"bookmarked": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- User Profile ----

@app.route('/api/profile', methods=['GET'])
@auth_required
def get_profile():
    """Get current user profile."""
    try:
        user = User.query.get(g.user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        return jsonify({
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "bio": user.bio,
            "avatar_url": user.avatar_url,
            "github_url": user.github_url,
            "twitter_url": user.twitter_url,
            "skills": user.skills,
            "created_at": user.created_at.isoformat(),
            "is_admin": user.email == os.getenv('ADMIN_EMAIL', 'abhijeet@mindinspo.local')
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/profile/<user_id>', methods=['GET'])
def get_public_profile(user_id):
    """Get a user's public profile and their ideas."""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Get published ideas by this user
        ideas = CatalogEntry.query.filter_by(user_id=user_id, visibility='public').order_by(CatalogEntry.created_at.desc()).all()
        ideas_data = []
        for idea in ideas:
            latest_update = IdeaUpdate.query.filter_by(catalog_entry_id=idea.id).order_by(IdeaUpdate.created_at.desc()).first()
            ideas_data.append({
                "id": idea.id,
                "raw_input": idea.raw_input,
                "input_type": idea.input_type,
                "summary": idea.summary,
                "image_url": idea.image_url,
                "tech_stack": idea.tech_stack,
                "idea_score": compute_idea_score(idea.id),
                "updates_count": IdeaUpdate.query.filter_by(catalog_entry_id=idea.id).count(),
                "latest_update": {
                    "content": latest_update.content,
                    "update_type": latest_update.update_type,
                    "created_at": latest_update.created_at.isoformat(),
                } if latest_update else None,
                "created_at": idea.created_at.isoformat()
            })

        # Calculate total reactions received on all public ideas
        total_reactions = {}
        for rtype in REACTION_TYPES:
            count = Reaction.query.join(CatalogEntry).filter(
                CatalogEntry.user_id == user_id, 
                Reaction.reaction_type == rtype,
                CatalogEntry.visibility == 'public'
            ).count()
            if count > 0:
                total_reactions[rtype] = count

        return jsonify({
            "id": user.id,
            "name": user.name or user.email.split('@')[0],
            "bio": user.bio,
            "avatar_url": user.avatar_url,
            "github_url": user.github_url,
            "twitter_url": user.twitter_url,
            "skills": user.skills,
            "created_at": user.created_at.isoformat(),
            "ideas": ideas_data,
            "published_count": len(ideas_data),
            "total_reactions": total_reactions
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/profile', methods=['PUT'])
@auth_required
def update_profile():
    """Update current user profile."""
    try:
        user = User.query.get(g.user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        data = request.json or {}
        if 'name' in data:
            user.name = data['name']
        if 'bio' in data:
            user.bio = data['bio']
        if 'avatar_url' in data:
            user.avatar_url = data['avatar_url']
        if 'github_url' in data:
            user.github_url = data['github_url']
        if 'twitter_url' in data:
            user.twitter_url = data['twitter_url']
        if 'skills' in data:
            user.skills = data['skills']

        db.session.commit()
        return jsonify({"status": "updated"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# ============================================================
# Admin API
# ============================================================

def admin_required(f):
    @wraps(f)
    @auth_required
    def decorated(*args, **kwargs):
        admin_email = os.getenv('ADMIN_EMAIL', 'abhijeet@mindinspo.local')
        if g.user_email != admin_email:
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def get_admin_stats():
    try:
        user_count = User.query.count()
        entry_count = CatalogEntry.query.count()
        idea_count = CatalogEntry.query.filter_by(input_type='idea').count()
        tool_count = CatalogEntry.query.filter_by(input_type='tool').count()
        
        # Check if 'visibility' column exists in CatalogEntry
        public_count = 0
        try:
            public_count = CatalogEntry.query.filter_by(visibility='public').count()
        except Exception as e:
            print(f"DEBUG: visibility count failed (maybe column missing): {e}", file=sys.stderr); sys.stderr.flush()

        comment_count = 0
        try:
            comment_count = Comment.query.count()
        except Exception as e:
            print(f"DEBUG: comment count failed: {e}", file=sys.stderr); sys.stderr.flush()
        
        print(f"DEBUG: Admin Stats - Users: {user_count}, Entries: {entry_count}", file=sys.stderr); sys.stderr.flush()
        
        return jsonify({
            "users": user_count,
            "entries": entry_count,
            "ideas": idea_count,
            "tools": tool_count,
            "public_entries": public_count,
            "comments": comment_count
        }), 200
    except Exception as e:
        print(f"ERROR in get_admin_stats: {e}", file=sys.stderr); sys.stderr.flush()
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_get_users():
    try:
        users = User.query.order_by(User.created_at.desc()).all()
        return jsonify([{
            "id": u.id,
            "email": u.email,
            "name": u.name,
            "created_at": u.created_at.isoformat()
        } for u in users]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/entries', methods=['GET'])
@admin_required
def admin_get_entries():
    try:
        entries = CatalogEntry.query.order_by(CatalogEntry.created_at.desc()).all()
        return jsonify([{
            "id": e.id,
            "user_id": e.user_id,
            "raw_input": e.raw_input,
            "status": e.status,
            "input_type": e.input_type,
            "visibility": e.visibility,
            "created_at": e.created_at.isoformat() if e.created_at else None
        } for e in entries]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Reactions API
# ============================================================

@app.route('/api/community/<entry_id>/react', methods=['POST'])
@auth_required
def toggle_reaction(entry_id):
    """Toggle a reaction on a public entry. Body: {reaction_type: 'brilliant'}"""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Entry not found"}), 404

        data = request.json or {}
        reaction_type = data.get('reaction_type', '').strip()
        if reaction_type not in REACTION_TYPES:
            return jsonify({"error": f"Invalid reaction_type. Must be one of: {REACTION_TYPES}"}), 400

        existing = Reaction.query.filter_by(
            catalog_entry_id=entry_id, user_id=g.user_id, reaction_type=reaction_type
        ).first()

        if existing:
            db.session.delete(existing)
            db.session.commit()
            toggled = False
        else:
            reaction = Reaction(
                id=str(uuid.uuid4()),
                catalog_entry_id=entry_id,
                user_id=g.user_id,
                reaction_type=reaction_type,
            )
            db.session.add(reaction)
            db.session.commit()
            toggled = True

            # Notify the post owner (only when adding, not removing)
            if entry.user_id != g.user_id:
                emoji_map = {'brilliant': '💡', 'interested': '👀', 'sellable': '💰', 'build_worthy': '🔨', 'needs_work': '🔧'}
                notif = Notification(
                    id=str(uuid.uuid4()),
                    user_id=entry.user_id,
                    type='reaction',
                    title=f"{g.user_name or g.user_email.split('@')[0]} reacted {emoji_map.get(reaction_type, '')} to your idea",
                    message=entry.raw_input[:100],
                    link=f'/community?post={entry_id}',
                )
                db.session.add(notif)
                db.session.commit()

        # Return updated counts
        reactions_summary = {}
        for rtype in REACTION_TYPES:
            reactions_summary[rtype] = Reaction.query.filter_by(catalog_entry_id=entry_id, reaction_type=rtype).count()

        user_reactions = [r.reaction_type for r in Reaction.query.filter_by(catalog_entry_id=entry_id, user_id=g.user_id).all()]

        return jsonify({
            "toggled": toggled,
            "reaction_type": reaction_type,
            "reactions": reactions_summary,
            "user_reactions": user_reactions,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Connect / Collaborate API
# ============================================================

@app.route('/api/community/<entry_id>/connect', methods=['POST'])
@auth_required
def send_connect_request(entry_id):
    """Send a collaboration request. Body: {role: 'developer', message: '...'}"""
    try:
        entry = CatalogEntry.query.get(entry_id)
        if not entry:
            return jsonify({"error": "Entry not found"}), 404
        if entry.user_id == g.user_id:
            return jsonify({"error": "Cannot connect with your own idea"}), 400

        data = request.json or {}
        role = data.get('role', '').strip()
        message = data.get('message', '').strip()

        if role not in CONNECT_ROLES:
            return jsonify({"error": f"Invalid role. Must be one of: {CONNECT_ROLES}"}), 400

        # Check if already sent this role
        existing = ConnectRequest.query.filter_by(
            catalog_entry_id=entry_id, requester_id=g.user_id, role=role
        ).first()
        if existing:
            return jsonify({"error": "You already sent a request for this role", "status": existing.status}), 409

        req = ConnectRequest(
            id=str(uuid.uuid4()),
            catalog_entry_id=entry_id,
            requester_id=g.user_id,
            owner_id=entry.user_id,
            role=role,
            message=message or None,
        )
        db.session.add(req)
        db.session.commit()

        # Notify the owner
        role_labels = {'co_founder': 'Co-Founder', 'developer': 'Developer', 'designer': 'Designer', 'advisor': 'Advisor', 'investor': 'Investor'}
        notif = Notification(
            id=str(uuid.uuid4()),
            user_id=entry.user_id,
            type='connect_request',
            title=f"{g.user_name or g.user_email.split('@')[0]} wants to collaborate as {role_labels.get(role, role)}",
            message=f'On: {entry.raw_input[:80]}' + (f' — "{message[:100]}"' if message else ''),
            link='/dashboard?tab=requests',
        )
        db.session.add(notif)
        db.session.commit()

        return jsonify({"status": "sent", "request_id": req.id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/connect/requests', methods=['GET'])
@auth_required
def get_incoming_requests():
    """Get collaboration requests received by the current user."""
    try:
        status_filter = request.args.get('status', 'all')
        query = ConnectRequest.query.filter_by(owner_id=g.user_id)
        if status_filter != 'all':
            query = query.filter_by(status=status_filter)
        reqs = query.order_by(ConnectRequest.created_at.desc()).all()

        result = []
        for r in reqs:
            requester = User.query.get(r.requester_id)
            entry = CatalogEntry.query.get(r.catalog_entry_id)
            result.append({
                "id": r.id,
                "role": r.role,
                "message": r.message,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
                "idea_title": entry.raw_input if entry else None,
                "idea_id": r.catalog_entry_id,
                "requester": {
                    "id": requester.id,
                    "name": requester.name or requester.email.split('@')[0],
                    "email": requester.email if r.status == 'accepted' else None,
                    "avatar_url": requester.avatar_url,
                    "github_url": requester.github_url if r.status == 'accepted' else None,
                    "twitter_url": requester.twitter_url if r.status == 'accepted' else None,
                } if requester else None,
            })
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/connect/sent', methods=['GET'])
@auth_required
def get_sent_requests():
    """Get collaboration requests sent by the current user."""
    try:
        reqs = ConnectRequest.query.filter_by(requester_id=g.user_id).order_by(ConnectRequest.created_at.desc()).all()
        result = []
        for r in reqs:
            owner = User.query.get(r.owner_id)
            entry = CatalogEntry.query.get(r.catalog_entry_id)
            result.append({
                "id": r.id,
                "role": r.role,
                "message": r.message,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
                "idea_title": entry.raw_input if entry else None,
                "idea_id": r.catalog_entry_id,
                "owner": {
                    "id": owner.id,
                    "name": owner.name or owner.email.split('@')[0],
                    "email": owner.email if r.status == 'accepted' else None,
                    "avatar_url": owner.avatar_url,
                    "github_url": owner.github_url if r.status == 'accepted' else None,
                    "twitter_url": owner.twitter_url if r.status == 'accepted' else None,
                } if owner else None,
            })
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/connect/requests/<request_id>/respond', methods=['POST'])
@auth_required
def respond_to_connect(request_id):
    """Accept or decline a collaboration request. Body: {action: 'accept'|'decline'}"""
    try:
        req = ConnectRequest.query.get(request_id)
        if not req:
            return jsonify({"error": "Request not found"}), 404
        if req.owner_id != g.user_id:
            return jsonify({"error": "Unauthorized"}), 403
        if req.status != 'pending':
            return jsonify({"error": "Request already responded to"}), 400

        data = request.json or {}
        action = data.get('action', '').strip()
        if action not in ('accept', 'decline'):
            return jsonify({"error": "Action must be 'accept' or 'decline'"}), 400

        req.status = 'accepted' if action == 'accept' else 'declined'
        req.updated_at = datetime.utcnow()
        
        if action == 'accept':
            # Add as a formal collaborator
            existing_collab = Collaborator.query.filter_by(user_id=req.requester_id, catalog_entry_id=req.catalog_entry_id).first()
            if not existing_collab:
                collab = Collaborator(
                    user_id=req.requester_id,
                    catalog_entry_id=req.catalog_entry_id,
                    role=req.role
                )
                db.session.add(collab)
        
        db.session.commit()

        # Notify the requester
        entry = CatalogEntry.query.get(req.catalog_entry_id)
        notif_type = 'connect_accepted' if action == 'accept' else 'connect_declined'
        notif_title = f"Your collaboration request was {'accepted' if action == 'accept' else 'declined'}!"
        notif = Notification(
            id=str(uuid.uuid4()),
            user_id=req.requester_id,
            type=notif_type,
            title=notif_title,
            message=f'For: {entry.raw_input[:80]}' if entry else '',
            link='/dashboard?tab=requests',
        )
        db.session.add(notif)
        db.session.commit()

        return jsonify({"status": req.status, "request_id": req.id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/connect/workspace/<request_id>', methods=['GET'])
@auth_required
def get_workspace(request_id):
    req = ConnectRequest.query.filter_by(id=request_id).first()
    if not req:
        return jsonify({'error': 'Not found'}), 404

    if g.user_id not in (req.requester_id, req.owner_id):
        return jsonify({'error': 'Forbidden'}), 403

    if req.status != 'accepted':
        return jsonify({'error': 'Request not accepted yet'}), 403

    entry = CatalogEntry.query.filter_by(id=req.catalog_entry_id).first()
    if not entry:
        return jsonify({'error': 'Idea not found'}), 404

    partner_id = req.requester_id if g.user_id == req.owner_id else req.owner_id
    partner = User.query.filter_by(id=partner_id).first()

    return jsonify({
        'request_id': req.id,
        'role': req.role,
        'status': req.status,
        'partner': {
            'id': partner.id,
            'name': partner.name or partner.email,
            'email': partner.email,
            'avatar_url': partner.avatar_url,
        },
        'entry': {
            'id': entry.id,
            'raw_input': entry.raw_input,
            'summary': entry.summary,
            'tech_stack': entry.tech_stack,
            'pros_cons': entry.pros_cons,
            'similar_tools': entry.similar_tools,
            'mermaid_syntax': entry.mermaid_syntax,
            'image_url': entry.image_url,
            'status': entry.status,
            'tags': entry.tags,
        },
    })


@app.route('/api/catalogs/<entry_id>/collaborators/<user_id>', methods=['DELETE'])
@auth_required
def remove_collaborator(entry_id, user_id):
    """Remove a collaborator from a project. Only the owner can do this."""
    try:
        entry = CatalogEntry.query.get_or_404(entry_id)
        if entry.user_id != g.user_id:
            return jsonify({"error": "Only the project owner can remove collaborators"}), 403
            
        collab = Collaborator.query.filter_by(catalog_entry_id=entry_id, user_id=user_id).first()
        if not collab:
            return jsonify({"error": "Collaborator not found"}), 404
            
        collab_name = collab.user.name or 'A collaborator'
        db.session.delete(collab)
        
        # Also add a system message to the chat
        try:
            sys_msg = ChatMessage(
                entry_id=entry_id,
                user_id=None,
                message=f"SYSTEM_MESSAGE::info::{collab_name} was removed from the team by the owner.",
                is_user=False
            )
            db.session.add(sys_msg)
        except Exception as e:
            print(f"Failed to add removal system message: {e}")

        db.session.commit()
        
        # Notify the user who was removed
        add_notification(user_id, f"You have been removed from the team for project '{entry.raw_input}'", "info")
        
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# ============================================================
# Notifications API
# ============================================================

@app.route('/api/notifications', methods=['GET'])
@auth_required
def get_notifications():
    """Get notifications for the current user."""
    try:
        limit = request.args.get('limit', 30, type=int)
        notifs = Notification.query.filter_by(user_id=g.user_id).order_by(
            Notification.created_at.desc()
        ).limit(limit).all()

        unread_count = Notification.query.filter_by(user_id=g.user_id, is_read=False).count()

        return jsonify({
            "notifications": [{
                "id": n.id,
                "type": n.type,
                "title": n.title,
                "message": n.message,
                "link": n.link,
                "is_read": n.is_read,
                "created_at": n.created_at.isoformat(),
            } for n in notifs],
            "unread_count": unread_count,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/notifications/read', methods=['POST'])
@auth_required
def mark_notifications_read():
    """Mark notifications as read. Body: {ids: [...]} or {all: true}"""
    try:
        data = request.json or {}
        if data.get('all'):
            Notification.query.filter_by(user_id=g.user_id, is_read=False).update({'is_read': True})
        else:
            ids = data.get('ids', [])
            if ids:
                Notification.query.filter(
                    Notification.id.in_(ids),
                    Notification.user_id == g.user_id
                ).update({'is_read': True}, synchronize_session='fetch')
        db.session.commit()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Production Startup Sequence (runs during Gunicorn import)
with app.app_context():
    try:
        init_db()

        # Start background thread
        retry_thread = threading.Thread(target=background_retry_job, args=(app,), daemon=True)
        retry_thread.start()
    except Exception as e:
        print(f"Startup sequence failed: {e}", file=sys.stderr)


if __name__ == '__main__':
    is_debug = os.getenv('FLASK_ENV', 'development').lower() == 'development'
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', '5000')),
        debug=is_debug,
        use_reloader=False,
    )
