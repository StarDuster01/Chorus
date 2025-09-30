#!/usr/bin/env python3
"""
Test script for image generation and variation functionality
"""

import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:50506"  # Change if different
# For VM: BASE_URL = "http://<your-vm-ip>:50506"

def get_auth_token(username="test", password="test"):
    """Login and get auth token"""
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": username, "password": password}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Logged in as {username}")
        return data.get("token")
    else:
        print(f"❌ Login failed: {response.status_code}")
        print(response.text)
        return None

def test_generate_new_image(token, bot_id, conversation_id=None):
    """Test generating a new image from scratch"""
    print("\n" + "="*60)
    print("TEST 1: Generate New Image from Scratch")
    print("="*60)
    
    message = "Generate a simple red circle on white background"
    
    response = requests.post(
        f"{BASE_URL}/api/bots/{bot_id}/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "message": message,
            "conversation_id": conversation_id,
            "debug_mode": True
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated new image")
        print(f"Response: {data.get('response', '')[:100]}...")
        
        if data.get('image_details'):
            img = data['image_details'][0]
            print(f"\nImage Details:")
            print(f"  URL: {img.get('url')}")
            print(f"  ID: {img.get('id')}")
            print(f"  Caption: {img.get('caption')}")
            return data.get('conversation_id'), img.get('id')
        else:
            print("⚠️  No image details in response")
            return data.get('conversation_id'), None
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)
        return None, None

def test_retrieve_image(token, bot_id, conversation_id):
    """Test retrieving an image from RAG database"""
    print("\n" + "="*60)
    print("TEST 2: Retrieve Image from Database")
    print("="*60)
    
    message = "Show me the kroger private selection logo"
    
    response = requests.post(
        f"{BASE_URL}/api/bots/{bot_id}/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "message": message,
            "conversation_id": conversation_id,
            "debug_mode": True
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Retrieved images")
        print(f"Response: {data.get('response', '')[:100]}...")
        
        if data.get('image_details'):
            for idx, img in enumerate(data['image_details'], 1):
                print(f"\nImage {idx}:")
                print(f"  Caption: {img.get('caption')}")
                print(f"  ID: {img.get('id')}")
                print(f"  URL: {img.get('url')}")
            
            # Return first image ID for variation testing
            return data.get('conversation_id'), data['image_details'][0].get('id')
        else:
            print("⚠️  No images found")
            return data.get('conversation_id'), None
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)
        return None, None

def test_variation_from_image(token, bot_id, image_id, conversation_id):
    """Test creating a variation from existing image"""
    print("\n" + "="*60)
    print("TEST 3: Create Variation from Existing Image")
    print("="*60)
    
    if not image_id:
        print("⚠️  No image ID provided, skipping test")
        return
    
    message = f"Generate from {image_id}: create a variation"
    print(f"Message: {message}")
    
    response = requests.post(
        f"{BASE_URL}/api/bots/{bot_id}/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "message": message,
            "conversation_id": conversation_id,
            "debug_mode": True
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Created variation")
        print(f"Response: {data.get('response', '')[:200]}...")
        
        if data.get('image_details'):
            img = data['image_details'][0]
            print(f"\nVariation Details:")
            print(f"  URL: {img.get('url')}")
            print(f"  New ID: {img.get('id')}")
            print(f"  Source ID: {img.get('source_image_id')}")
            print(f"  Caption: {img.get('caption')}")
        
        if data.get('debug'):
            debug = data['debug']
            print(f"\nDebug Info:")
            print(f"  Mode: {debug.get('mode')}")
            print(f"  Source ID: {debug.get('source_image_id')}")
            print(f"  Source Caption: {debug.get('source_image_caption')}")
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)

def main():
    print("\n🧪 Image Generation Test Suite\n")
    
    # Get credentials from command line or use defaults
    username = sys.argv[1] if len(sys.argv) > 1 else "test"
    password = sys.argv[2] if len(sys.argv) > 2 else "test"
    bot_id = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Login
    token = get_auth_token(username, password)
    if not token:
        print("❌ Cannot proceed without authentication")
        return
    
    # Get or request bot ID
    if not bot_id:
        bot_id = input("\nEnter Bot ID to test with: ").strip()
        if not bot_id:
            print("❌ Bot ID is required")
            return
    
    conversation_id = None
    
    # Test 1: Generate new image
    conversation_id, generated_image_id = test_generate_new_image(token, bot_id, conversation_id)
    
    # Test 2: Retrieve image from database
    conversation_id, retrieved_image_id = test_retrieve_image(token, bot_id, conversation_id)
    
    # Test 3: Create variation from retrieved image
    if retrieved_image_id:
        test_variation_from_image(token, bot_id, retrieved_image_id, conversation_id)
    else:
        print("\n⚠️  Skipping variation test - no image found in database")
        print("   Upload some images to a dataset first!")
    
    print("\n" + "="*60)
    print("✅ Test Suite Complete")
    print("="*60)
    print(f"\nConversation ID: {conversation_id}")
    print(f"Check the conversation in the UI to see all generated images")

if __name__ == "__main__":
    print("Usage: python test_image_generation.py [username] [password] [bot_id]")
    main()
