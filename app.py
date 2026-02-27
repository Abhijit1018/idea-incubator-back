import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from models import db, User, CatalogEntry
from datetime import datetime
import uuid
import requests
import json
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Use SQLite for local development ease, but can switch to Postgres via DATABASE_URL
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///incubator.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

N8N_WEBHOOK_URL = os.getenv('N8N_WEBHOOK_URL', 'http://localhost:5678/webhook/idea-incubator')

with app.app_context():
    db.create_all()

@app.route('/api/ideas/submit', methods=['POST'])
def submit_idea():
    data = request.json
    raw_input = data.get('raw_input')
    
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
        status='pending'
    )
    db.session.add(new_entry)
    db.session.commit()

    # Trigger n8n Webhook asynchronously (in reality, use a task queue like Celery, but here we do a simple non-blocking request or just blocking since it's a prototype)
    # To prevent blocking, we just fire and forget if possible, or wait with a short timeout.
    try:
        # We pass the entry_id so n8n can callback and update the record
        requests.post(N8N_WEBHOOK_URL, json={"entry_id": entry_id, "raw_input": raw_input}, timeout=5.0)
    except requests.exceptions.ReadTimeout:
        pass # Expected since we don't wait for response
    except requests.exceptions.RequestException as e:
        print(f"Error triggering n8n webhook: {e}")

    return jsonify({"message": "Idea submitted successfully", "entry_id": entry_id, "status": "pending"}), 201

@app.route('/api/catalogs/', methods=['GET'])
def get_catalogs():
    entries = CatalogEntry.query.order_by(CatalogEntry.created_at.desc()).all()
    result = []
    for e in entries:
        result.append({
            "id": e.id,
            "user_id": e.user_id,
            "raw_input": e.raw_input,
            "status": e.status,
            "summary": e.summary,
            "tech_stack": e.tech_stack,
            "pros_cons": e.pros_cons,
            "similar_tools": e.similar_tools,
            "mermaid_syntax": e.mermaid_syntax,
            "image_url": e.image_url,
            "created_at": e.created_at.isoformat(),
        })
    return jsonify(result), 200

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
            image_url = f"http://localhost:5000/static/uploads/{unique_filename}"

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

        # 6. Return a standard JSON response
        return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"Webhook processing error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
