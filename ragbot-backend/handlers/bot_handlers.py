import os
import json
import uuid
import datetime
from datetime import UTC
from flask import request, jsonify
from handlers.dataset_handlers import find_dataset_by_id
from constants import DATASETS_FOLDER

# Bot handler functions
def get_bots_handler(user_data):
    """Get all bots for a user
    
    Args:
        user_data: User data from JWT token
    
    Returns:
        tuple: JSON response and status code
    """
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    os.makedirs(bots_dir, exist_ok=True)
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify([]), 200
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
    
    return jsonify(bots), 200

def create_bot_handler(user_data):
    """Create a new bot
    
    Args:
        user_data: User data from JWT token
    
    Returns:
        tuple: JSON response and status code
    """
    data = request.json
    
    if not data or not data.get('name'):
        return jsonify({"error": "Bot name is required"}), 400
    
    # Process dataset IDs - can be added later or supplied in the request
    dataset_ids = []
    if data.get('dataset_id'):  # Support single dataset_id in request for backward compatibility
        dataset_ids.append(data.get('dataset_id'))
    if data.get('dataset_ids') and isinstance(data.get('dataset_ids'), list):
        dataset_ids.extend(data.get('dataset_ids'))
    
    # Get chorus_id if provided
    chorus_id = data.get('chorus_id', '')
    
    # Verify that the chorus exists if one was provided
    if chorus_id:
        chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
        user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
        
        if os.path.exists(user_choruses_file):
            with open(user_choruses_file, 'r') as f:
                choruses = json.load(f)
                
            # Find the chorus by ID
            chorus = next((c for c in choruses if c["id"] == chorus_id), None)
            if not chorus:
                return jsonify({"error": f"Chorus with ID {chorus_id} not found"}), 400
    
    # Create a new bot
    new_bot = {
        "id": str(uuid.uuid4()),
        "name": data.get('name'),
        "description": data.get('description', ''),
        "dataset_ids": dataset_ids,
        "chorus_id": chorus_id,  # Set the chorus ID
        "prompt_template": data.get('prompt_template', ''),
        "system_instruction": data.get('system_instruction', 'You are a helpful AI assistant. Answer questions based on the provided context.'),
        "created_at": datetime.datetime.now(UTC).isoformat()
    }
    
    # Save the bot
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    os.makedirs(bots_dir, exist_ok=True)
    
    # Check if user already has bots
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if os.path.exists(user_bots_file):
        with open(user_bots_file, 'r') as f:
            bots = json.load(f)
        bots.append(new_bot)
    else:
        bots = [new_bot]
    
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
    
    return jsonify(new_bot), 201

def delete_bot_handler(user_data, bot_id):
    """Delete a bot
    
    Args:
        user_data: User data from JWT token
        bot_id: ID of the bot to delete
    
    Returns:
        tuple: JSON response and status code
    """
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
    
    # Find and remove the bot
    bot_found = False
    for i, bot in enumerate(bots):
        if bot["id"] == bot_id:
            bots.pop(i)
            bot_found = True
            break
    
    if not bot_found:
        return jsonify({"error": "Bot not found"}), 404
    
    # Save updated bots list
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
    
    return jsonify({"message": "Bot deleted successfully"}), 200

def get_bot_datasets_handler(user_data, bot_id):
    """Get datasets associated with a bot
    
    Args:
        user_data: User data from JWT token
        bot_id: ID of the bot
    
    Returns:
        tuple: JSON response and status code
    """
    # Get bot info
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
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
    
    # Get dataset info for each dataset ID
    datasets_dir = DATASETS_FOLDER
    
    dataset_details = []
    
    # Get dataset IDs from bot
    dataset_ids = bot.get("dataset_ids", [])
    
    # First try to find datasets in the user's own datasets
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    user_datasets = []
    
    if os.path.exists(user_datasets_file):
        try:
            with open(user_datasets_file, 'r') as f:
                user_datasets = json.load(f)
        except Exception as e:
            print(f"Error loading user datasets: {str(e)}")
    
    # Check all dataset files to find missing datasets
    # This searches across all users' datasets
    for dataset_id in dataset_ids:
        # First check if it's in the user's own datasets
        dataset_found = False
        
        for d in user_datasets:
            if d["id"] == dataset_id:
                dataset_details.append(d)
                dataset_found = True
                break
                
        if dataset_found:
            continue
            
        # If not found in user's datasets, search all dataset files
        # Note: This is a cross-user search which may not be desired for security
        # For now, we'll skip this and mark as missing
        dataset = None
        if dataset:
            dataset_details.append(dataset)
        else:
            # Add placeholder for truly missing dataset
            dataset_details.append({
                "id": dataset_id,
                "name": "Unknown dataset",
                "missing": True
            })
    
    # Get all available datasets for the selection dropdown
    all_available_datasets = user_datasets.copy()
    
    # Add a flag to show which datasets are already associated with this bot
    for d in all_available_datasets:
        d["is_associated"] = d["id"] in dataset_ids
    
    return jsonify({
        "bot": bot,
        "datasets": dataset_details,
        "available_datasets": all_available_datasets
    }), 200

def add_dataset_to_bot_handler(user_data, bot_id):
    """Add a dataset to a bot
    
    Args:
        user_data: User data from JWT token
        bot_id: ID of the bot
    
    Returns:
        tuple: JSON response and status code
    """
    data = request.json
    dataset_id = data.get('dataset_id')
    
    if not dataset_id:
        return jsonify({"error": "Dataset ID is required"}), 400
    
    # Check if dataset exists
    datasets_dir = DATASETS_FOLDER
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    dataset_exists = False
    dataset_name = "Unknown dataset"
    if os.path.exists(user_datasets_file):
        with open(user_datasets_file, 'r') as f:
            datasets = json.load(f)
        
        for d in datasets:
            if d["id"] == dataset_id:
                dataset_exists = True
                dataset_name = d["name"]
                break
    
    if not dataset_exists:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Get bot info
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    bot_index = None
    for i, b in enumerate(bots):
        if b["id"] == bot_id:
            bot_index = i
            break
            
    if bot_index is None:
        return jsonify({"error": "Bot not found"}), 404
    
    # Add dataset to bot
    if "dataset_ids" not in bots[bot_index]:
        bots[bot_index]["dataset_ids"] = []
        
    # Check if dataset is already associated with this bot
    if dataset_id in bots[bot_index]["dataset_ids"]:
        return jsonify({"message": f"Dataset '{dataset_name}' is already associated with this bot"}), 200
        
    bots[bot_index]["dataset_ids"].append(dataset_id)
    
    # Save updated bots
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
    
    return jsonify({"message": f"Dataset '{dataset_name}' added to bot successfully"}), 200

def remove_dataset_from_bot_handler(user_data, bot_id, dataset_id):
    """Remove a dataset from a bot
    
    Args:
        user_data: User data from JWT token
        bot_id: ID of the bot
        dataset_id: ID of the dataset to remove
    
    Returns:
        tuple: JSON response and status code
    """
    # Get bot info
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    # Find the bot
    bot_index = None
    for i, b in enumerate(bots):
        if b["id"] == bot_id:
            bot_index = i
            break
            
    if bot_index is None:
        return jsonify({"error": "Bot not found"}), 404
    
    # Check if dataset is associated with this bot
    if "dataset_ids" not in bots[bot_index] or dataset_id not in bots[bot_index]["dataset_ids"]:
        return jsonify({"error": "Dataset is not associated with this bot"}), 404
        
    # Remove dataset from bot
    bots[bot_index]["dataset_ids"].remove(dataset_id)
    
    # Save updated bots
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
    
    return jsonify({"message": "Dataset removed from bot successfully"}), 200

def set_bot_datasets_handler(user_data, bot_id):
    """Set all datasets for a bot (replace existing ones)
    
    Args:
        user_data: User data from JWT token
        bot_id: ID of the bot
    
    Returns:
        tuple: JSON response and status code
    """
    data = request.json
    dataset_ids = data.get('dataset_ids', [])
    
    if not isinstance(dataset_ids, list):
        return jsonify({"error": "dataset_ids must be an array"}), 400
    
    # Get bot info
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    # Find the bot
    bot_found = False
    for bot in bots:
        if bot["id"] == bot_id:
            bot["dataset_ids"] = dataset_ids
            bot_found = True
            break
            
    if not bot_found:
        return jsonify({"error": "Bot not found"}), 404
    
    # Save updated bots
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
    
    return jsonify({"message": "Bot datasets updated successfully"}), 200 