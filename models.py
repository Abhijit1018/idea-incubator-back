from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()

def generate_uuid():
    return str(uuid.uuid4())

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class CatalogEntry(db.Model):
    __tablename__ = 'catalog_entries'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    raw_input = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(50), default='pending') # pending, completed, failed
    input_type = db.Column(db.String(50), default='idea') # idea or tool
    retry_count = db.Column(db.Integer, default=0)
    summary = db.Column(db.Text, nullable=True)
    tech_stack = db.Column(db.JSON, nullable=True)
    pros_cons = db.Column(db.JSON, nullable=True)
    similar_tools = db.Column(db.JSON, nullable=True)
    mermaid_syntax = db.Column(db.Text, nullable=True)
    image_url = db.Column(db.String(500), nullable=True)
    creator = db.Column(db.String(255), nullable=True)
    link = db.Column(db.String(500), nullable=True)
    installation = db.Column(db.Text, nullable=True)
    unique_features = db.Column(db.JSON, nullable=True)
    market_trend = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    catalog_entry_id = db.Column(db.String(36), db.ForeignKey('catalog_entries.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, default=True) # True if message is from user, False if from AI
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    catalog_entry = db.relationship('CatalogEntry', backref=db.backref('chat_messages', lazy=True))
    user = db.relationship('User', backref=db.backref('chat_messages', lazy=True))


class CatalogEmbedding(db.Model):
    __tablename__ = 'catalog_embeddings'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    catalog_entry_id = db.Column(db.String(36), db.ForeignKey('catalog_entries.id'), nullable=False)
    # Using pgvector extension for vector storage
    # We'll store the embedding as a vector (assuming 1536 dimensions for OpenAI embeddings)
    embedding = db.Column(db.PickleType, nullable=False)  # Using PickleType for simplicity, can change to Vector later
    # Alternative: Store as JSON and handle vector conversion in service layer
    embedding_json = db.Column(db.JSON, nullable=True)  # For compatibility
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    catalog_entry = db.relationship('CatalogEntry', backref=db.backref('embeddings', lazy=True))
