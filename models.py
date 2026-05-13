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
    name = db.Column(db.String(120), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    avatar_url = db.Column(db.String(500), nullable=True)
    github_url = db.Column(db.String(500), nullable=True)
    twitter_url = db.Column(db.String(500), nullable=True)
    skills = db.Column(db.JSON, nullable=True)
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
    # Community fields
    visibility = db.Column(db.String(20), default='private')  # private, public
    published_at = db.Column(db.DateTime, nullable=True)
    view_count = db.Column(db.Integer, default=0)
    tags = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = db.relationship('User', backref=db.backref('catalog_entries', lazy=True))


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


class Comment(db.Model):
    __tablename__ = 'comments'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    catalog_entry_id = db.Column(db.String(36), db.ForeignKey('catalog_entries.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    parent_id = db.Column(db.String(36), db.ForeignKey('comments.id'), nullable=True)  # For threading
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    catalog_entry = db.relationship('CatalogEntry', backref=db.backref('comments', lazy=True, order_by='Comment.created_at'))
    user = db.relationship('User', backref=db.backref('comments', lazy=True))
    replies = db.relationship('Comment', backref=db.backref('parent', remote_side=[id]), lazy=True)


class Like(db.Model):
    __tablename__ = 'likes'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    catalog_entry_id = db.Column(db.String(36), db.ForeignKey('catalog_entries.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Unique constraint: one like per user per entry
    __table_args__ = (db.UniqueConstraint('catalog_entry_id', 'user_id', name='uq_like_user_entry'),)

    # Relationships
    catalog_entry = db.relationship('CatalogEntry', backref=db.backref('likes', lazy=True))
    user = db.relationship('User', backref=db.backref('likes', lazy=True))


class Bookmark(db.Model):
    __tablename__ = 'bookmarks'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    catalog_entry_id = db.Column(db.String(36), db.ForeignKey('catalog_entries.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('catalog_entry_id', 'user_id', name='uq_bookmark_user_entry'),)

    catalog_entry = db.relationship('CatalogEntry', backref=db.backref('bookmarks', lazy=True))
    user = db.relationship('User', backref=db.backref('bookmarks', lazy=True))


# Valid reaction types
REACTION_TYPES = ['brilliant', 'interested', 'sellable', 'build_worthy', 'needs_work']

# Valid connect roles
CONNECT_ROLES = ['co_founder', 'developer', 'designer', 'advisor', 'investor']


class Reaction(db.Model):
    __tablename__ = 'reactions'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    catalog_entry_id = db.Column(db.String(36), db.ForeignKey('catalog_entries.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    reaction_type = db.Column(db.String(20), nullable=False)  # brilliant, interested, sellable, build_worthy, needs_work
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # One reaction per type per user per entry
    __table_args__ = (db.UniqueConstraint('catalog_entry_id', 'user_id', 'reaction_type', name='uq_reaction_user_entry_type'),)

    catalog_entry = db.relationship('CatalogEntry', backref=db.backref('reactions', lazy=True))
    user = db.relationship('User', backref=db.backref('reactions', lazy=True))


class ConnectRequest(db.Model):
    __tablename__ = 'connect_requests'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    catalog_entry_id = db.Column(db.String(36), db.ForeignKey('catalog_entries.id'), nullable=False)
    requester_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    owner_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    role = db.Column(db.String(30), nullable=False)  # co_founder, developer, designer, advisor, investor
    message = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='pending')  # pending, accepted, declined
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # One request per role per user per entry
    __table_args__ = (db.UniqueConstraint('catalog_entry_id', 'requester_id', 'role', name='uq_connect_requester_entry_role'),)

    catalog_entry = db.relationship('CatalogEntry', backref=db.backref('connect_requests', lazy=True))
    requester = db.relationship('User', foreign_keys=[requester_id], backref=db.backref('sent_requests', lazy=True))
    owner = db.relationship('User', foreign_keys=[owner_id], backref=db.backref('received_requests', lazy=True))


class Notification(db.Model):
    __tablename__ = 'notifications'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    type = db.Column(db.String(30), nullable=False)  # reaction, connect_request, connect_accepted, connect_declined, comment
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=True)
    link = db.Column(db.String(500), nullable=True)  # frontend route to navigate to
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('notifications', lazy=True, order_by='Notification.created_at.desc()'))


class IdeaUpdate(db.Model):
    __tablename__ = 'idea_updates'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    catalog_entry_id = db.Column(db.String(36), db.ForeignKey('catalog_entries.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    update_type = db.Column(db.String(30), default='progress')  # progress, feedback, changelog, milestone
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    catalog_entry = db.relationship('CatalogEntry', backref=db.backref('updates', lazy=True, order_by='IdeaUpdate.created_at.desc()'))
    user = db.relationship('User', backref=db.backref('idea_updates', lazy=True))
