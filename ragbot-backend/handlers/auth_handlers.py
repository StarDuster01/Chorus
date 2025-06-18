import os
import json
import uuid
import datetime
from datetime import UTC
import bcrypt
import jwt
import functools
from flask import request, jsonify
from constants import USERS_FOLDER

# Helper functions for authentication
def get_token_from_header():
    """Extract token from Authorization header
    
    Returns:
        str: Token or None if not found
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header or 'Bearer ' not in auth_header:
        return None
    return auth_header.split('Bearer ')[1]

def verify_token(jwt_secret_key):
    """Verify the JWT token
    
    Args:
        jwt_secret_key: Secret key for JWT verification
    
    Returns:
        dict: Decoded token data or None if invalid
    """
    token = get_token_from_header()
    if not token:
        return None
    try:
        decoded = jwt.decode(token, jwt_secret_key, algorithms=["HS256"])
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(jwt_secret_key):
    """Decorator for routes that require authentication
    
    Args:
        jwt_secret_key: Secret key for JWT verification
    
    Returns:
        function: Decorator function
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            user_data = verify_token(jwt_secret_key)
            if not user_data:
                return jsonify({"error": "Unauthorized"}), 401
            return f(user_data, *args, **kwargs)
        return decorated
    return functools.wraps(decorator)(decorator)

# Authentication handlers
def register_handler(jwt_secret_key, jwt_expires):
    """Register a new user
    
    Args:
        jwt_secret_key: Secret key for JWT
        jwt_expires: Token expiration time
    
    Returns:
        tuple: JSON response and status code
    """
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
        
    # Check if user already exists - for demo purposes using a simple file
    users_file = os.path.join(USERS_FOLDER, "users.json")
    users = {}
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            users = json.load(f)
    
    if username in users:
        return jsonify({"error": "Username already exists"}), 400
        
    # Hash password and store user
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user_id = str(uuid.uuid4())
    users[username] = {
        "id": user_id,
        "username": username,
        "password": hashed_password
    }
    
    with open(users_file, 'w') as f:
        json.dump(users, f)
        
    # Generate token
    token = jwt.encode(
        {"id": user_id, "username": username, "exp": datetime.datetime.now(UTC) + jwt_expires},
        jwt_secret_key
    )
    
    return jsonify({"token": token, "user": {"id": user_id, "username": username}}), 201

def login_handler(jwt_secret_key, jwt_expires):
    """Login an existing user
    
    Args:
        jwt_secret_key: Secret key for JWT
        jwt_expires: Token expiration time
    
    Returns:
        tuple: JSON response and status code
    """
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
        
    # Check user credentials - for demo purposes using a simple file
    users_file = os.path.join(USERS_FOLDER, "users.json")
    
    if not os.path.exists(users_file):
        return jsonify({"error": "Invalid username or password"}), 401
        
    with open(users_file, 'r') as f:
        users = json.load(f)
        
    if username not in users:
        return jsonify({"error": "Invalid username or password"}), 401
        
    user = users[username]
    if not bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
        return jsonify({"error": "Invalid username or password"}), 401
        
    # Generate token
    token = jwt.encode(
        {"id": user["id"], "username": username, "exp": datetime.datetime.now(UTC) + jwt_expires},
        jwt_secret_key
    )
    
    return jsonify({"token": token, "user": {"id": user["id"], "username": username}}), 200 