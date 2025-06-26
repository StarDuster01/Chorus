import os
import json
import uuid
import datetime
from datetime import UTC
from flask import jsonify, request

from constants import CHORUSES_FOLDER, BOTS_FOLDER

# Helper functions for chorus management
def _get_default_choruses():
    """Loads the default chorus configurations."""
    default_choruses = []
    # Adjusted path to be relative to the script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_choruses_file = os.path.join(base_dir, '..', 'default_choruses.json')
    if os.path.exists(default_choruses_file):
        try:
            with open(default_choruses_file, 'r') as f:
                default_choruses = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading default choruses: {e}", flush=True)
    return default_choruses

def _get_user_choruses(user_id):
    """Loads a user's specific chorus configurations."""
    user_choruses_file = os.path.join(CHORUSES_FOLDER, f"{user_id}_choruses.json")
    if not os.path.exists(user_choruses_file):
        return []
    try:
        with open(user_choruses_file, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return []

def _save_user_choruses(user_id, choruses):
    """Saves a user's chorus configurations."""
    user_choruses_file = os.path.join(CHORUSES_FOLDER, f"{user_id}_choruses.json")
    with open(user_choruses_file, 'w') as f:
        json.dump(choruses, f, indent=2)

def get_chorus_config_handler(user_data, bot_id):
    user_bots_file = os.path.join(BOTS_FOLDER, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    bot = next((b for b in bots if b["id"] == bot_id), None)
    if not bot:
        return jsonify({"error": "Bot not found"}), 404
    
    chorus_id = bot.get("chorus_id")
    if not chorus_id:
        return jsonify({"error": "No chorus configuration found for this bot"}), 404
    
    # Check both default and user choruses
    all_choruses = _get_default_choruses() + _get_user_choruses(user_data['id'])
    chorus = next((c for c in all_choruses if c["id"] == chorus_id), None)
    
    if not chorus:
        return jsonify({"error": f"Chorus configuration with ID {chorus_id} not found"}), 404
    
    return jsonify(chorus), 200

def save_chorus_config_handler(user_data, bot_id):
    # This endpoint is now simplified to just associate a chorus with a bot
    # For backward compatibility
    
    # Check if bot exists and belongs to user
    # Using BOTS_FOLDER from constants
    user_bots_file = os.path.join(BOTS_FOLDER, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    bot = None
    bot_index = -1
    for i, b in enumerate(bots):
        if b["id"] == bot_id:
            bot = b
            bot_index = i
            break
            
    if not bot:
        return jsonify({"error": "Bot not found"}), 404
    
    # Get the chorus data
    data = request.json
    
    # First, create or update a chorus
    try:
        # Create chorus directory if it doesn't exist
        # Using CHORUSES_FOLDER from constants
        
        # Load or create user's chorus list
        user_choruses = _get_user_choruses(user_data['id'])
        
        # Generate a unique ID for the chorus
        chorus_id = str(uuid.uuid4())
        
        # Create chorus data
        chorus_data = {
            "id": chorus_id,
            "name": data.get('name', 'Unnamed Chorus'),
            "description": data.get('description', ''),
            "response_models": data.get('response_models', []),
            "evaluator_models": data.get('evaluator_models', []),
            "use_diverse_rag": data.get('use_diverse_rag', False),  # Add the diverse RAG setting
            "created_at": datetime.datetime.now(UTC).isoformat(),
            "updated_at": datetime.datetime.now(UTC).isoformat(),
            "created_by": user_data['username']
        }
        
        # Add the chorus
        user_choruses.append(chorus_data)
        
        # Save the updated choruses
        _save_user_choruses(user_data['id'], user_choruses)
            
        # Update the bot to use this chorus
        bots[bot_index]["chorus_id"] = chorus_id
        
        # Save the updated bot
        with open(user_bots_file, 'w') as f:
            json.dump(bots, f)
        
        return jsonify({
            "message": "Chorus created and assigned to bot successfully", 
            "config": chorus_data,
            "bot": bots[bot_index]
        }), 200
    except Exception as e:
        print(f"Error creating chorus: {str(e)}")
        return jsonify({"error": f"Error creating chorus: {str(e)}"}), 500

def set_bot_chorus_handler(user_data, bot_id):
    # Check if bot exists and belongs to user
    # Using BOTS_FOLDER from constants
    user_bots_file = os.path.join(BOTS_FOLDER, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    bot = None
    bot_index = -1
    for i, b in enumerate(bots):
        if b["id"] == bot_id:
            bot = b
            bot_index = i
            break
            
    if not bot:
        return jsonify({"error": "Bot not found"}), 404
    
    # Get the chorus ID from the request
    data = request.json
    chorus_id = data.get('chorus_id', '')
    
    # Empty string means unassign any chorus
    if chorus_id:
        # Verify that the chorus exists if an ID was provided
        all_choruses = _get_default_choruses() + _get_user_choruses(user_data['id'])
        chorus = next((c for c in all_choruses if c["id"] == chorus_id), None)
        if not chorus:
            return jsonify({"error": f"Chorus with ID {chorus_id} not found"}), 404
    
    # Update the bot with the new chorus ID
    bots[bot_index]["chorus_id"] = chorus_id
    
    # Save the updated bot
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
    
    return jsonify({
        "message": chorus_id and "Chorus assigned to bot successfully" or "Chorus unassigned from bot successfully", 
        "bot": bots[bot_index]
    }), 200

def list_choruses_handler(user_data):
    """Returns a list of all available choruses (default + user-specific)."""
    default_choruses = _get_default_choruses()
    user_choruses = _get_user_choruses(user_data['id'])
    
    # Ensure the user chorus file exists, creating it if necessary
    user_choruses_file = os.path.join(CHORUSES_FOLDER, f"{user_data['id']}_choruses.json")
    if not os.path.exists(user_choruses_file):
        _save_user_choruses(user_data['id'], [])

    # Combine lists, ensuring no duplicates from defaults
    final_choruses = default_choruses + user_choruses
    
    return jsonify(final_choruses), 200

def create_chorus_handler(user_data):
    # Get chorus data from request
    data = request.json
    
    # Basic validation
    if not data.get('name'):
        return jsonify({"error": "Chorus name is required"}), 400
        
    if not data.get('response_models') or not isinstance(data.get('response_models'), list) or len(data.get('response_models')) == 0:
        return jsonify({"error": "At least one response model is required"}), 400
        
    if not data.get('evaluator_models') or not isinstance(data.get('evaluator_models'), list) or len(data.get('evaluator_models')) == 0:
        return jsonify({"error": "At least one evaluator model is required"}), 400
    
    user_choruses = _get_user_choruses(user_data['id'])
    
    # Generate a unique ID for the chorus
    chorus_id = str(uuid.uuid4())
    
    # Add the new chorus with metadata
    chorus = {
        "id": chorus_id,
        "name": data.get('name'),
        "description": data.get('description', ''),
        "created_at": datetime.datetime.now(UTC).isoformat(),
        "updated_at": datetime.datetime.now(UTC).isoformat(),
        "response_models": data.get('response_models', []),
        "evaluator_models": data.get('evaluator_models', []),
        "use_diverse_rag": data.get('use_diverse_rag', False),
        "created_by": user_data['username']
    }
    
    user_choruses.append(chorus)
    
    # Save updated chorus list
    _save_user_choruses(user_data['id'], user_choruses)
    
    return jsonify(chorus), 201

def get_chorus_handler(user_data, chorus_id):
    """Gets a single chorus by its ID, checking both default and user choruses."""
    all_choruses = _get_default_choruses() + _get_user_choruses(user_data['id'])
    chorus = next((c for c in all_choruses if c["id"] == chorus_id), None)
    
    if not chorus:
        return jsonify({"error": "Chorus not found"}), 404
    
    return jsonify(chorus), 200

def update_chorus_handler(user_data, chorus_id):
    # Prevent updating default choruses
    if any(c['id'] == chorus_id for c in _get_default_choruses()):
        return jsonify({"error": "Default choruses cannot be modified."}), 403
        
    # Get chorus data from request
    data = request.json
    
    # Basic validation
    if not data.get('name'):
        return jsonify({"error": "Chorus name is required"}), 400
        
    if not data.get('response_models') or not isinstance(data.get('response_models'), list) or len(data.get('response_models')) == 0:
        return jsonify({"error": "At least one response model is required"}), 400
        
    if not data.get('evaluator_models') or not isinstance(data.get('evaluator_models'), list) or len(data.get('evaluator_models')) == 0:
        return jsonify({"error": "At least one evaluator model is required"}), 400
    
    user_choruses = _get_user_choruses(user_data['id'])
    
    # Find and update the chorus
    chorus_index = next((i for i, c in enumerate(user_choruses) if c["id"] == chorus_id), None)
    if chorus_index is None:
        return jsonify({"error": "Chorus not found"}), 404
    
    # Update the chorus with new data, preserving the ID and creation date
    chorus = user_choruses[chorus_index]
    updated_chorus = {
        "id": chorus["id"],
        "name": data.get('name'),
        "description": data.get('description', ''),
        "created_at": chorus["created_at"],
        "updated_at": datetime.datetime.now(UTC).isoformat(),
        "response_models": data.get('response_models', []),
        "evaluator_models": data.get('evaluator_models', []),
        "use_diverse_rag": data.get('use_diverse_rag', False),
        "created_by": chorus.get("created_by", user_data['username'])
    }
    
    user_choruses[chorus_index] = updated_chorus
    
    # Save updated chorus list
    _save_user_choruses(user_data['id'], user_choruses)
    
    return jsonify(updated_chorus), 200

def delete_chorus_handler(user_data, chorus_id):
    # Prevent deleting default choruses
    if any(c['id'] == chorus_id for c in _get_default_choruses()):
        return jsonify({"error": "Default choruses cannot be deleted."}), 403
        
    user_choruses = _get_user_choruses(user_data['id'])
    
    # Find and remove the chorus
    original_length = len(user_choruses)
    user_choruses = [c for c in user_choruses if c["id"] != chorus_id]
    
    if len(user_choruses) == original_length:
        return jsonify({"error": "Chorus not found"}), 404
    
    # Save updated chorus list
    _save_user_choruses(user_data['id'], user_choruses)
    
    # Also check if any bots are using this chorus and update them
    # Using BOTS_FOLDER from constants
    user_bots_file = os.path.join(BOTS_FOLDER, f"{user_data['id']}_bots.json")
    
    if os.path.exists(user_bots_file):
        with open(user_bots_file, 'r') as f:
            bots = json.load(f)
        
        # Update any bots using this chorus
        updated = False
        for bot in bots:
            if bot.get("chorus_id") == chorus_id:
                bot["chorus_id"] = ""
                updated = True
        
        # Save updated bots if needed
        if updated:
            with open(user_bots_file, 'w') as f:
                json.dump(bots, f)
    
    return jsonify({"message": "Chorus deleted successfully"}), 200 
