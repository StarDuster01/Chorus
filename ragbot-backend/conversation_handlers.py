import os
import json
import datetime
from datetime import UTC
import uuid
from flask import jsonify

def get_conversations_handler(user_data, bot_id, conversations_folder):
    """Get all conversations for a specific bot"""
    try:
        # Read all conversation files for this user and bot
        conversation_files = [f for f in os.listdir(conversations_folder) 
                             if f.startswith(f"{user_data['id']}_{bot_id}_") and f.endswith('.json')]
        
        conversations = []
        for file_name in conversation_files:
            file_path = os.path.join(conversations_folder, file_name)
            try:
                with open(file_path, 'r') as f:
                    conversation = json.load(f)
                
                # Add a preview of the conversation
                if conversation.get("messages"):
                    # Get the first user message as title if not already set
                    if not conversation.get("title"):
                        first_user_message = next((msg for msg in conversation["messages"] if msg["role"] == "user"), None)
                        if first_user_message:
                            title = first_user_message["content"]
                            conversation["title"] = title[:40] + "..." if len(title) > 40 else title
                    
                    # Count messages
                    message_count = len(conversation["messages"])
                    conversation["message_count"] = message_count
                
                # Add a summary of the conversation
                conversations.append({
                    "id": conversation["id"],
                    "title": conversation.get("title", "Untitled Conversation"),
                    "created_at": conversation.get("created_at"),
                    "updated_at": conversation.get("updated_at"),
                    "message_count": conversation.get("message_count", 0),
                    "preview": conversation["messages"][-1]["content"][:100] + "..." if conversation.get("messages") else ""
                })
            except Exception as e:
                print(f"Error loading conversation {file_name}: {str(e)}")
        
        # Sort conversations by updated_at (newest first)
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return jsonify(conversations), 200
    except Exception as e:
        print(f"Error getting conversations: {str(e)}")
        return jsonify({"error": f"Failed to retrieve conversations: {str(e)}"}), 500

def get_conversation_handler(user_data, bot_id, conversation_id, conversations_folder):
    """Get a specific conversation with all messages"""
    try:
        # Check if conversation exists
        conversation_file = os.path.join(conversations_folder, f"{user_data['id']}_{bot_id}_{conversation_id}.json")
        if not os.path.exists(conversation_file):
            return jsonify({"error": "Conversation not found"}), 404
        
        # Read conversation
        with open(conversation_file, 'r') as f:
            conversation = json.load(f)
        
        return jsonify(conversation), 200
    except Exception as e:
        print(f"Error getting conversation: {str(e)}")
        return jsonify({"error": f"Failed to retrieve conversation: {str(e)}"}), 500

def delete_conversation_handler(user_data, bot_id, conversation_id, conversations_folder):
    """Delete a specific conversation"""
    try:
        # Check if conversation exists
        conversation_file = os.path.join(conversations_folder, f"{user_data['id']}_{bot_id}_{conversation_id}.json")
        if not os.path.exists(conversation_file):
            return jsonify({"error": "Conversation not found"}), 404
        
        # Delete the file
        os.remove(conversation_file)
        
        return jsonify({"message": "Conversation deleted successfully"}), 200
    except Exception as e:
        print(f"Error deleting conversation: {str(e)}")
        return jsonify({"error": f"Failed to delete conversation: {str(e)}"}), 500

def delete_all_conversations_handler(user_data, bot_id, conversations_folder):
    """Delete all conversations for a specific bot"""
    try:
        # Find all conversation files for this user and bot
        conversation_files = [f for f in os.listdir(conversations_folder) 
                             if f.startswith(f"{user_data['id']}_{bot_id}_") and f.endswith('.json')]
        
        # Delete each file
        for file_name in conversation_files:
            file_path = os.path.join(conversations_folder, file_name)
            os.remove(file_path)
        
        return jsonify({"message": f"Deleted {len(conversation_files)} conversations successfully"}), 200
    except Exception as e:
        print(f"Error deleting all conversations: {str(e)}")
        return jsonify({"error": f"Failed to delete conversations: {str(e)}"}), 500

def rename_conversation_handler(user_data, bot_id, conversation_id, new_title, conversations_folder):
    """Rename a conversation"""
    try:
        if not new_title:
            return jsonify({"error": "Title is required"}), 400
        
        # Check if conversation exists
        conversation_file = os.path.join(conversations_folder, f"{user_data['id']}_{bot_id}_{conversation_id}.json")
        if not os.path.exists(conversation_file):
            return jsonify({"error": "Conversation not found"}), 404
        
        # Read conversation
        with open(conversation_file, 'r') as f:
            conversation = json.load(f)
        
        # Update title
        conversation["title"] = new_title
        
        # Save updated conversation
        with open(conversation_file, 'w') as f:
            json.dump(conversation, f)
        
        return jsonify({"message": "Conversation renamed successfully", "title": new_title}), 200
    except Exception as e:
        print(f"Error renaming conversation: {str(e)}")
        return jsonify({"error": f"Failed to rename conversation: {str(e)}"}), 500 