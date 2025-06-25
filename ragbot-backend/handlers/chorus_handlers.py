import os
import json
import uuid
import datetime
from datetime import UTC
from flask import jsonify, request

from constants import CHORUSES_FOLDER, BOTS_FOLDER

def get_chorus_config_handler(user_data, bot_id):
    # Check if bot exists and belongs to user
    # Using BOTS_FOLDER from constants
    user_bots_file = os.path.join(BOTS_FOLDER, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    bot = None
    for b in bots:
        if b["id"] == bot_id:
            bot = b
            break
            
    if not bot:
        return jsonify({"error": "Bot not found"}), 404
    
    # Check for chorus_id in the bot
    chorus_id = bot.get("chorus_id", "")
    if not chorus_id:
        return jsonify({"error": "No chorus configuration found for this bot"}), 404
    
    # Get the user's chorus definitions
    # Using CHORUSES_FOLDER from constants
    
    user_choruses_file = os.path.join(CHORUSES_FOLDER, f"{user_data['id']}_choruses.json")
    if not os.path.exists(user_choruses_file):
        return jsonify({"error": "No chorus configurations found"}), 404
    
    # Load the user's choruses and find the one with the matching ID
    with open(user_choruses_file, 'r') as f:
        choruses = json.load(f)
    
    chorus = next((c for c in choruses if c["id"] == chorus_id), None)
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
        user_choruses_file = os.path.join(CHORUSES_FOLDER, f"{user_data['id']}_choruses.json")
        
        if os.path.exists(user_choruses_file):
            with open(user_choruses_file, 'r') as f:
                choruses = json.load(f)
        else:
            choruses = []
        
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
        choruses.append(chorus_data)
        
        # Save the updated choruses
        with open(user_choruses_file, 'w') as f:
            json.dump(choruses, f)
            
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
        # Using CHORUSES_FOLDER from constants
        user_choruses_file = os.path.join(CHORUSES_FOLDER, f"{user_data['id']}_choruses.json")
        
        if os.path.exists(user_choruses_file):
            with open(user_choruses_file, 'r') as f:
                choruses = json.load(f)
                
            # Find the chorus by ID
            chorus = next((c for c in choruses if c["id"] == chorus_id), None)
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
    # Get all chorus configurations
    # Using CHORUSES_FOLDER from constants
    
    # First check if there's a user-specific choruses file
    user_choruses_file = os.path.join(CHORUSES_FOLDER, f"{user_data['id']}_choruses.json")
    
    # If the user choruses file doesn't exist yet, create it
    if not os.path.exists(user_choruses_file):
        with open(user_choruses_file, 'w') as f:
            json.dump([], f)
    
    # Load user's chorus definitions
    with open(user_choruses_file, 'r') as f:
        choruses = json.load(f)
    
    return jsonify(choruses), 200

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
    
    # Create chorus directory if it doesn't exist
    # Using CHORUSES_FOLDER from constants
    
    # Load user's chorus definitions
    user_choruses_file = os.path.join(CHORUSES_FOLDER, f"{user_data['id']}_choruses.json")
    if os.path.exists(user_choruses_file):
        with open(user_choruses_file, 'r') as f:
            choruses = json.load(f)
    else:
        choruses = []
    
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
    
    choruses.append(chorus)
    
    # Save updated chorus list
    with open(user_choruses_file, 'w') as f:
        json.dump(choruses, f)
    
    return jsonify(chorus), 201

def get_chorus_handler(user_data, chorus_id):
    # Get chorus directory and user choruses file
    # Using CHORUSES_FOLDER from constants
    user_choruses_file = os.path.join(CHORUSES_FOLDER, f"{user_data['id']}_choruses.json")
    
    if not os.path.exists(user_choruses_file):
        return jsonify({"error": "Chorus not found"}), 404
    
    # Load user's chorus definitions
    with open(user_choruses_file, 'r') as f:
        choruses = json.load(f)
    
    # Find the requested chorus
    chorus = next((c for c in choruses if c["id"] == chorus_id), None)
    if not chorus:
        return jsonify({"error": "Chorus not found"}), 404
    
    return jsonify(chorus), 200

def update_chorus_handler(user_data, chorus_id):
    # Get chorus data from request
    data = request.json
    
    # Basic validation
    if not data.get('name'):
        return jsonify({"error": "Chorus name is required"}), 400
        
    if not data.get('response_models') or not isinstance(data.get('response_models'), list) or len(data.get('response_models')) == 0:
        return jsonify({"error": "At least one response model is required"}), 400
        
    if not data.get('evaluator_models') or not isinstance(data.get('evaluator_models'), list) or len(data.get('evaluator_models')) == 0:
        return jsonify({"error": "At least one evaluator model is required"}), 400
    
    # Get chorus directory and user choruses file
    # Using CHORUSES_FOLDER from constants
    user_choruses_file = os.path.join(CHORUSES_FOLDER, f"{user_data['id']}_choruses.json")
    
    if not os.path.exists(user_choruses_file):
        return jsonify({"error": "Chorus not found"}), 404
    
    # Load user's chorus definitions
    with open(user_choruses_file, 'r') as f:
        choruses = json.load(f)
    
    # Find and update the chorus
    chorus_index = next((i for i, c in enumerate(choruses) if c["id"] == chorus_id), None)
    if chorus_index is None:
        return jsonify({"error": "Chorus not found"}), 404
    
    # Update the chorus with new data, preserving the ID and creation date
    chorus = choruses[chorus_index]
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
    
    choruses[chorus_index] = updated_chorus
    
    # Save updated chorus list
    with open(user_choruses_file, 'w') as f:
        json.dump(choruses, f)
    
    return jsonify(updated_chorus), 200

def delete_chorus_handler(user_data, chorus_id):
    # Get chorus directory and user choruses file
    # Using CHORUSES_FOLDER from constants
    user_choruses_file = os.path.join(CHORUSES_FOLDER, f"{user_data['id']}_choruses.json")
    
    if not os.path.exists(user_choruses_file):
        return jsonify({"error": "Chorus not found"}), 404
    
    # Load user's chorus definitions
    with open(user_choruses_file, 'r') as f:
        choruses = json.load(f)
    
    # Find and remove the chorus
    original_length = len(choruses)
    choruses = [c for c in choruses if c["id"] != chorus_id]
    
    if len(choruses) == original_length:
        return jsonify({"error": "Chorus not found"}), 404
    
    # Save updated chorus list
    with open(user_choruses_file, 'w') as f:
        json.dump(choruses, f)
    
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
