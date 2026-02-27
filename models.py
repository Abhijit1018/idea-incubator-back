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
    summary = db.Column(db.Text, nullable=True)
    tech_stack = db.Column(db.JSON, nullable=True)
    pros_cons = db.Column(db.JSON, nullable=True)
    similar_tools = db.Column(db.JSON, nullable=True)
    mermaid_syntax = db.Column(db.Text, nullable=True)
    image_url = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
