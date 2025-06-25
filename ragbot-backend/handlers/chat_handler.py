from flask import request, jsonify
import os
import json
import uuid
import base64
import re
import traceback
import datetime
import openai
import anthropic
import requests

# your ChromaDB client and image‐RAG processor
import chroma_client
from image_processor import image_processor

# helpers you actually call in this handler
from handlers.image_handlers import resize_image
from handlers.dataset_handlers import find_dataset_by_id
from constants import BASE_DIR
from constants import IMAGE_FOLDER
from constants import CONVERSATIONS_FOLDER
from constants import DEFAULT_LLM_MODEL
from constants import IMAGE_GENERATION_MODEL
from constants import DATASETS_FOLDER

# timezone used for timestamping
UTC = datetime.timezone.utc

# Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
#

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "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"))
# ——————————————————————————————————————————————
#


def chat_with_bot_handler(user_data, bot_id):
    data = request.json
    message = data.get('message', '')
    debug_mode = data.get('debug_mode', False)
    use_model_chorus = data.get('use_model_chorus', False)  # User's explicit choice to use model chorus
    chorus_id = data.get('chorus_id', '')  # A specific chorus ID to use
    conversation_id = data.get('conversation_id', '')  # The conversation this message belongs to
    
    # Check for image data in base64 format
    image_data = data.get('image_data', '')
    image_path = None
    has_image = False
    
    if not message and not image_data:
        return jsonify({"error": "Message or image is required"}), 400
    
    # Create a new conversation if no conversation_id provided
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Use AI to intelligently detect if this is an image generation request
    is_image_generation_request = False
    image_generation_prompt = ""
    
    # Get conversation context for better analysis
    conversation_context = ""
    if os.path.exists(conversation_file):
        with open(conversation_file, 'r') as f:
            existing_conversation = json.load(f)
            recent_messages = existing_conversation.get("messages", [])[-5:]  # Last 5 messages for context
            conversation_context = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_messages 
                if isinstance(msg['content'], str) and msg['role'] in ['user', 'assistant']
            ])
    
    # Use AI to determine if the user wants an image generated (vs retrieving existing images from dataset)
    try:
        intent_analysis_prompt = f"""Analyze if the user is requesting a NEW image to be generated/created/drawn, versus asking to see EXISTING images from a knowledge base/dataset.

Conversation context:
{conversation_context}

Current user message: "{message}"

Respond with a JSON object in this exact format:
{{
    "is_image_request": true/false,
    "image_description": "detailed description of what image to generate (only if is_image_request is true)"
}}

GENERATE NEW IMAGE (is_image_request: true):
- "Can you show me what a sunset looks like?" (general/creative request)
- "I'd like to see a cat" (generic animal/object)
- "Draw me something beautiful" (creative request)
- "What would a futuristic city look like?" (hypothetical/creative)
- "Make an image of a forest" (explicit generation)
- "Generate a picture of..." (explicit generation)
- "Create an illustration of..." (explicit creation)
- "What does happiness look like?" (abstract concept)

RETRIEVE EXISTING IMAGES (is_image_request: false):
- "Show me our products" (company-specific)
- "What does our logo look like?" (company-specific)
- "Show me the diagram from the presentation" (document-specific)
- "Display our company policy images" (organization-specific)
- "Show me the screenshots" (dataset-specific)
- "What images do you have about X?" (asking about existing content)
- "Find pictures of our events" (organization-specific)

Key distinction: If the request is about specific organizational content, existing documents, or company-specific material, choose FALSE. If it's a general creative request or hypothetical visualization, choose TRUE.

Only respond with the JSON object, nothing else."""

        intent_response = openai.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing user intent for image generation requests. Always respond with valid JSON only."},
                {"role": "user", "content": intent_analysis_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Parse the AI response
        try:
            intent_result = json.loads(intent_response.choices[0].message.content.strip())
            is_image_generation_request = intent_result.get("is_image_request", False)
            if is_image_generation_request:
                image_generation_prompt = intent_result.get("image_description", "")
                print(f"AI detected image request: {is_image_generation_request}, prompt: {image_generation_prompt}", flush=True)
        except json.JSONDecodeError:
            print(f"Failed to parse intent analysis response: {intent_response.choices[0].message.content}", flush=True)
            # Fall back to conservative keyword detection (avoid conflicts with dataset image retrieval)
            message_lower = message.lower().strip()
            # More specific keywords that clearly indicate NEW image generation (not dataset retrieval)
            generation_keywords = ["draw me", "create an image", "generate an image", "make an image", "create a picture", "generate a picture"]
            is_image_generation_request = any(keyword in message_lower for keyword in generation_keywords)
            if is_image_generation_request:
                image_generation_prompt = message  # Use original message as fallback
    
    except Exception as intent_error:
        print(f"Error in intent analysis: {str(intent_error)}", flush=True)
        # Fall back to conservative keyword detection (avoid conflicts with dataset image retrieval)
        message_lower = message.lower().strip()
        # More specific keywords that clearly indicate NEW image generation (not dataset retrieval)
        generation_keywords = ["draw me", "create an image", "generate an image", "make an image", "create a picture", "generate a picture"]
        is_image_generation_request = any(keyword in message_lower for keyword in generation_keywords)
        if is_image_generation_request:
            image_generation_prompt = message  # Use original message as fallback
    
    # If image is provided, save it to a temporary file
    if image_data:
        has_image = True
        try:
            # Extract the base64 content and file extension
            if ';base64,' in image_data:
                header, encoded = image_data.split(';base64,')
                file_ext = header.split('/')[-1]
            else:
                encoded = image_data
                file_ext = 'png'  # Default to PNG if not specified
            
            # Decode the base64 data
            decoded_image = base64.b64decode(encoded)
            
            # Save to a temporary file
            filename = f"chat_image_{str(uuid.uuid4())}.{file_ext}"
            image_path = os.path.join(IMAGE_FOLDER, filename)
            
            with open(image_path, 'wb') as f:
                f.write(decoded_image)
                
            # Resize image if needed - use the max_dimension parameter
            image_path = resize_image(image_path, max_dimension=1024)
            
            # Update message to include reference to the image
            if not message:
                message = "[Image uploaded]"
            else:
                message = f"{message} [Image uploaded]"
                
        except Exception as img_error:
            print(f"Error processing image: {str(img_error)}", flush=True)
            return jsonify({"error": f"Failed to process image: {str(img_error)}"}), 400
    
    # Create a new message object
    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": message,
        "timestamp": datetime.datetime.now(UTC).isoformat()
    }
    
    # If there's an image, add it to the message metadata
    if has_image:
        user_message["has_image"] = True
        user_message["image_path"] = image_path
    
    # Add message to conversation history
    conversation_file = os.path.join(CONVERSATIONS_FOLDER, f"{user_data['id']}_{bot_id}_{conversation_id}.json")
    conversation_exists = os.path.exists(conversation_file)
    
    if conversation_exists:
        # Load existing conversation
        with open(conversation_file, 'r') as f:
            conversation = json.load(f)
    else:
        # Create new conversation
        conversation = {
            "id": conversation_id,
            "bot_id": bot_id,
            "user_id": user_data['id'],
            "title": message[:40] + "..." if len(message) > 40 else message,  # Use first message as title
            "created_at": datetime.datetime.now(UTC).isoformat(),
            "updated_at": datetime.datetime.now(UTC).isoformat(),
            "messages": []
        }
    
    # Add user message to conversation
    conversation["messages"].append(user_message)
    conversation["updated_at"] = datetime.datetime.now(UTC).isoformat()
    
    # Save updated conversation
    with open(conversation_file, 'w') as f:
        json.dump(conversation, f)
    
    # Handle image generation requests
    if is_image_generation_request:
        try:
            # If no specific prompt, use the original message
            if not image_generation_prompt:
                image_generation_prompt = message
            
            # Enhance the image prompt for better generation
            enhancement_system_message = """You are an expert at creating detailed image prompts for AI image generation. 
Enhance the user's request by adding visual details, artistic style, lighting, mood, and composition while keeping the core intent.
Make it specific and visually compelling. Respond with ONLY the enhanced prompt text, nothing else."""

            enhance_response = openai.chat.completions.create(
                model=DEFAULT_LLM_MODEL,
                messages=[
                    {"role": "system", "content": enhancement_system_message},
                    {"role": "user", "content": f"Enhance this image request: {image_generation_prompt}"}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            enhanced_prompt = enhance_response.choices[0].message.content.strip()
            
            # First, respond to the user about what we're going to generate
            bot_response_content = f"I'll create an image for you! I'm generating: \"{enhanced_prompt}\""
            
            print(f"Generating image with enhanced prompt: {enhanced_prompt}", flush=True)
            
            # Generate the image using existing image generation logic
            generation_params = {
                "model": "gpt-image-1",
                "prompt": enhanced_prompt,
                "n": 1,
                "size": "1024x1024",
                "quality": "auto",
                "moderation": "auto"
            }
            
            # Generate image with OpenAI
            image_response = openai.images.generate(**generation_params)
            
            if image_response.data:
                image_obj = image_response.data[0]
                
                # Get the base64 content from OpenAI response
                if hasattr(image_obj, 'b64_json') and image_obj.b64_json:
                    image_content = base64.b64decode(image_obj.b64_json)
                    image_url = None
                elif hasattr(image_obj, 'url') and image_obj.url:
                    image_url = image_obj.url
                    # Download the image
                    image_download_response = requests.get(image_url)
                    if image_download_response.status_code != 200:
                        raise Exception("Failed to download generated image")
                    image_content = image_download_response.content
                else:
                    raise Exception("No image data found in response")
                
                # Save the generated image
                filename = f"generated_{str(uuid.uuid4())}.png"
                filepath = os.path.join(IMAGE_FOLDER, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_content)
                
                # Create the URL for accessing the image
                api_image_url = f"/api/images/{filename}"
                
                # Create bot response with the generated image
                bot_response = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant", 
                    "content": bot_response_content,
                    "timestamp": datetime.datetime.now(UTC).isoformat(),
                    "generated_image": True,
                    "image_url": api_image_url,
                    "image_prompt": enhanced_prompt
                }
                
                conversation["messages"].append(bot_response)
                with open(conversation_file, 'w') as f:
                    json.dump(conversation, f)
                
                # Return response with image details
                image_details = [{
                    "index": "Generated Image",
                    "caption": f"Generated: {enhanced_prompt}",
                    "url": api_image_url,
                    "download_url": f"{api_image_url}?download=true",
                    "id": str(uuid.uuid4()),
                    "dataset_id": ""
                }]
                
                return jsonify({
                    "response": bot_response_content,
                    "conversation_id": conversation_id,
                    "image_generated": True,
                    "image_details": image_details,
                    "image_prompt_used": enhanced_prompt,
                    "debug": {"original_prompt": image_generation_prompt, "enhanced_prompt": enhanced_prompt} if debug_mode else None
                }), 200
            else:
                raise Exception("No image was generated")
                
        except Exception as img_gen_error:
            print(f"Error generating image: {str(img_gen_error)}", flush=True)
            # If image generation fails, fall back to regular chat response
            error_response = f"I tried to generate an image for you, but encountered an error: {str(img_gen_error)}. Let me help you with a text response instead."
            
            bot_response = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": error_response,
                "timestamp": datetime.datetime.now(UTC).isoformat(),
                "image_generation_failed": True
            }
            conversation["messages"].append(bot_response)
            with open(conversation_file, 'w') as f:
                json.dump(conversation, f)
            
            return jsonify({
                "response": error_response,
                "conversation_id": conversation_id,
                "image_generation_failed": True,
                "debug": {"error": str(img_gen_error)} if debug_mode else None
            }), 200
        
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
        
    # Get dataset IDs from the bot
    dataset_ids = bot.get("dataset_ids", [])
    
    # Process image with OpenAI Vision API if an image is present
    if has_image:
        try:
            # Read the image file and encode it as base64
            with open(image_path, 'rb') as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Prepare the message with the image for OpenAI Vision API
            messages = [
                {
                    "role": "system", 
                    "content": bot.get("system_instruction", "You are a helpful assistant that can analyze images.")
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": message if message != "[Image uploaded]" else "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{file_ext};base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Call OpenAI API with vision capabilities
            response = openai.chat.completions.create(
                model=DEFAULT_LLM_MODEL,
                messages=messages,
                max_tokens=1024
            )
            
            response_text = response.choices[0].message.content
            
            # Save the response in the conversation
            bot_response = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.datetime.now(UTC).isoformat(),
                "from_image_analysis": True
            }
            conversation["messages"].append(bot_response)
            with open(conversation_file, 'w') as f:
                json.dump(conversation, f)
                
            return jsonify({
                "response": response_text,
                "conversation_id": conversation_id,
                "image_processed": True
            }), 200
            
        except Exception as vision_error:
            print(f"Error processing image with Vision API: {str(vision_error)}", flush=True)
            # If the vision API fails, continue with the regular RAG process
    
    if not dataset_ids:
        return jsonify({
            "response": "I don't have any datasets to work with. Please add a dataset to help me answer your questions.",
            "conversation_id": conversation_id
        }), 200
    
    # Check if the bot has a chorus configuration associated with it
    # If so, use model chorus by default unless explicitly turned off
    bot_has_chorus = bot.get("chorus_id", "")
    use_model_chorus = use_model_chorus or bool(bot_has_chorus)
    
    # If a specific chorus_id is provided, use it; otherwise use the bot's chorus_id if available
    specific_chorus_id = chorus_id or bot.get("chorus_id", "")
    
    # Retrieve relevant documents from all datasets
    all_contexts = []
    image_results = []
    
    try:
        for dataset_id in dataset_ids:
            try:
                # Text-based document retrieval
                collection = chroma_client.get_collection(dataset_id)
                
                # Check if collection has any documents
                collection_count = collection.count()
                if collection_count > 0:
                    # Determine how many results to request based on collection size
                    n_results = min(15, collection_count)  # Increased for better coverage
                    
                    # First get a sample to check if we're dealing with PowerPoint content
                    sample_results = collection.query(
                        query_texts=[message],
                        n_results=1
                    )
                    
                    # Check if the sample contains PowerPoint content
                    is_powerpoint = False
                    if sample_results and sample_results["documents"] and sample_results["documents"][0]:
                        sample_text = sample_results["documents"][0][0]
                        if "## SLIDE " in sample_text or "## PRESENTATION SUMMARY ##" in sample_text:
                            is_powerpoint = True
                            # For PowerPoint content, get more chunks to ensure we have enough context
                            n_results = min(20, collection_count)  # More chunks for PowerPoint content
                    
                    # Get the actual results
                    results = collection.query(
                        query_texts=[message],
                        n_results=n_results
                    )
                    
                    # Debug: log retrieved contexts
                    if results["documents"][0]:
                        print(f"CONTEXT DEBUG: Retrieved {len(results['documents'][0])} chunks from dataset {dataset_id}")
                        for i, chunk in enumerate(results["documents"][0][:10]):  # Show first 5 chunks for better debugging
                            preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                            print(f"CONTEXT DEBUG: Chunk {i+1}: {preview}")
                            # Check for key terms
                            if "private selection" in chunk.lower():
                                print(f"CONTEXT DEBUG: ✅ Chunk {i+1} contains Private Selection data")
                            if "simple truth" in chunk.lower():
                                print(f"CONTEXT DEBUG: ✅ Chunk {i+1} contains Simple Truth data")
                    else:
                        print(f"CONTEXT DEBUG: No chunks retrieved from dataset {dataset_id}")
                    
                    all_contexts.extend(results["documents"][0])
                
                # Image-based retrieval if available
                try:
                    # First, check if dataset contains images
                    has_images = False
                    dataset_type = "text"  # Default
                    
                    # Enhanced image query detection
                    image_query_terms = [
                        "image", "picture", "photo", "visual", "diagram", "graph", 
                        "chart", "illustration", "screenshot", "scan", "drawing", 
                        "artwork", "logo", "icon", "figure", "graphic", "view", 
                        "show me", "look like", "appearance", "visual", "display",
                        "find a picture", "find image", "show image", "display the visual",
                        "include the diagram", "include image", "with image", "any image",
                        "the picture", "the image", "the logo", "the diagram", "the illustration"
                    ]
                    
                    # Use a more aggressive check for image queries - partial matches and phrases
                    is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
                    
                    # Debug: Log image query detection decision
                    print(f"IMAGE SEARCH DEBUG: Query '{message}' - Is image query? {is_image_query}", flush=True)
                    if is_image_query:
                        matching_terms = [term for term in image_query_terms if term in message.lower()]
                        print(f"IMAGE SEARCH DEBUG: Matched terms: {matching_terms}", flush=True)
                    
                    # Check dataset type and image count from metadata first
                    datasets_dir = DATASETS_FOLDER
                    for user_id in [user_data['id'], "shared"]:  # Check both user and shared datasets
                        user_datasets_file = os.path.join(datasets_dir, f"{user_id}_datasets.json")
                        if os.path.exists(user_datasets_file):
                            with open(user_datasets_file, 'r') as f:
                                datasets = json.load(f)
                                
                                for ds in datasets:
                                    if ds["id"] == dataset_id:
                                        dataset_type = ds.get("type", "text")
                                        if dataset_type in ["image", "mixed"] and ds.get("image_count", 0) > 0:
                                            has_images = True
                                        break
                    
                    # Direct check in image_processor metadata to confirm images exist
                    dataset_has_metadata = dataset_id in image_processor.image_metadata
                    metadata_count = len(image_processor.image_metadata.get(dataset_id, [])) if dataset_has_metadata else 0
                    
                    # Skip image retrieval if this is not an image query AND dataset has no images
                    # But always mark dataset as having data if images exist, even for non-image queries
                    if not is_image_query and not (has_images or metadata_count > 0):
                        print(f"IMAGE SEARCH DEBUG: Skipping image retrieval for non-image query with no images: '{message}'", flush=True)
                        continue
                    
                    # If dataset has images according to either source, retrieve them
                    if has_images or metadata_count > 0:
                        print(f"Dataset {dataset_id} has images: dataset says {has_images}, metadata has {metadata_count}", flush=True)
                        
                        # Set search parameters based on whether we're using chorus mode and query type
                        if is_image_query:
                            # For explicit image queries, search more aggressively
                            if use_model_chorus:
                                top_k = 8
                            else:
                                top_k = 6
                        else:
                            # For non-image queries, just grab a few images to ensure bot knows it has data
                            # But don't search with the user's query - use generic search instead
                            if use_model_chorus:
                                top_k = 2  # Just enough to show the bot has data
                            else:
                                top_k = 1  # Minimal for standard mode
                        
                        print(f"IMAGE SEARCH DEBUG: Will search for up to {top_k} images (is_image_query: {is_image_query})", flush=True)
                        
                        # Search for images - use different strategy based on query type
                        img_results = []
                        try:
                            if is_image_query:
                                # For image queries, search with the user's specific query
                                print(f"IMAGE SEARCH DEBUG: Starting semantic image search for '{message}'", flush=True)
                                img_results = image_processor.search_images(dataset_id, message, top_k=top_k)
                                print(f"IMAGE SEARCH DEBUG: Found {len(img_results)} images for query '{message}' in dataset {dataset_id}", flush=True)
                            else:
                                # For non-image queries, just get some representative images to show the bot has data
                                print(f"IMAGE SEARCH DEBUG: Getting representative images for non-image query", flush=True)
                                # Use a very generic query to get any available images
                                generic_query = "image"
                                img_results = image_processor.search_images(dataset_id, generic_query, top_k=top_k)
                                print(f"IMAGE SEARCH DEBUG: Found {len(img_results)} representative images in dataset {dataset_id}", flush=True)
                                
                            if img_results:
                                for i, img in enumerate(img_results):
                                    print(f"IMAGE SEARCH DEBUG: Image {i+1}: {img.get('caption', 'No caption')} (score: {img.get('score', 0):.4f})", flush=True)
                        except Exception as img_search_error:
                            print(f"IMAGE SEARCH ERROR: {str(img_search_error)}", flush=True)
                        
                        # If no results or weak results, also try a more generic search, but only for image queries
                        if (not img_results or (img_results and img_results[0].get("score", 0) < 0.2)) and is_image_query:
                            # Try searching with a more generic query 
                            generic_queries = [
                                "relevant image for this topic",
                                "visual representation",
                                "image related to this subject",
                                "picture about this"
                            ]
                            
                            # Add more specific generic queries for likely image questions
                            if is_image_query:
                                generic_queries.extend([
                                    "show me images about this",
                                    "find relevant visuals",
                                    "diagrams or images for this topic",
                                    "picture showing this",
                                    "visual of this information"
                                ])
                                
                            # For chorus mode, be more aggressive in finding images
                            if use_model_chorus:
                                generic_queries.extend([
                                    "important image",
                                    "key visual",
                                    "significant illustration",
                                    "main diagram",
                                    "helpful picture for reference"
                                ])
                            
                            print(f"IMAGE SEARCH DEBUG: Using fallback generic queries: {generic_queries[:3]}...", flush=True)
                            
                            for generic_query in generic_queries:
                                try:
                                    print(f"IMAGE SEARCH DEBUG: Trying generic query: '{generic_query}'", flush=True)
                                    generic_results = image_processor.search_images(dataset_id, generic_query, top_k=4 if is_image_query else 2)
                                    if generic_results:
                                        for gen_img in generic_results:
                                            # Add to results if not already included
                                            if not any(img.get("id") == gen_img.get("id") for img in img_results):
                                                img_results.append(gen_img)
                                        if len(img_results) >= top_k:
                                            break  # Stop after finding enough results
                                except Exception as generic_search_error:
                                    print(f"Error with generic image search: {str(generic_search_error)}", flush=True)
                                    continue
                        
                        # Add relevant images to results
                        if img_results:
                            for img in img_results:
                                # Add image information to results
                                image_info = {
                                    "id": img["id"],
                                    "caption": img["caption"],
                                    "url": f"/api/images/{os.path.basename(img['path'])}",
                                    "score": img["score"],
                                    "dataset_id": dataset_id
                                }
                                image_results.append(image_info)
                                
                except Exception as img_error:
                    print(f"Error with image retrieval for dataset '{dataset_id}': {str(img_error)}", flush=True)
                    # Continue even if image retrieval fails
                
            except Exception as coll_error:
                print(f"Error with collection '{dataset_id}': {str(coll_error)}", flush=True)
                # Continue with other datasets even if one fails
                continue
                
        if not all_contexts and not image_results:
            # Save the response in the conversation
            bot_response = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": "I don't have any documents or images in my knowledge base yet. Please upload some content to help me answer your questions.",
                "timestamp": datetime.datetime.now(UTC).isoformat()
            }
            conversation["messages"].append(bot_response)
            with open(conversation_file, 'w') as f:
                json.dump(conversation, f)
                
            return jsonify({
                "response": "I don't have any documents or images in my knowledge base yet. Please upload some content to help me answer your questions.",
                "debug": {"error": "No contexts found in any collections"} if debug_mode else None,
                "conversation_id": conversation_id
            }), 200
            
        # Sort contexts by relevance (they should already be sorted from query results)
        # But we need to truncate to avoid token limits
        
        # Check if we have PowerPoint content in the retrieved contexts
        has_powerpoint = any("## SLIDE " in ctx or "## PRESENTATION SUMMARY ##" in ctx for ctx in all_contexts)
        
        # Adjust the maximum number of contexts based on content type
        if has_powerpoint:
            max_contexts = 15  # Allow more contexts for PowerPoint content
        else:
            max_contexts = 12  # Increased limit for regular content
            
        contexts = all_contexts[:max_contexts]
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        # Debug: log final context summary
        print(f"FINAL CONTEXT DEBUG: Using {len(contexts)} total chunks (max_contexts: {max_contexts})")
        if contexts:
            print(f"FINAL CONTEXT DEBUG: Total context length: {len(context_text)} characters")
            print(f"FINAL CONTEXT DEBUG: First context preview: {contexts[0][:300]}..." if len(contexts[0]) > 300 else f"FINAL CONTEXT DEBUG: First context: {contexts[0]}")
        else:
            print("FINAL CONTEXT DEBUG: No contexts available for query")
        
        # Prepare image information for context
        image_context = ""
        dataset_id_to_name = {}
        if image_results:
            # Build a mapping from dataset_id to dataset name
            for img in image_results:
                dsid = img["dataset_id"]
                if dsid not in dataset_id_to_name:
                    ds, _ = find_dataset_by_id(user_data, dsid)
                    if ds and ds.get("name"):
                        dataset_id_to_name[dsid] = ds["name"]
                    else:
                        dataset_id_to_name[dsid] = dsid[:8]  # fallback to short id
            # Sort images by score
            image_results.sort(key=lambda x: x["score"], reverse=True)
            image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
            is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
            max_images = 5 if is_image_query else 3
            top_images = image_results[:max_images]
            # If more than one dataset, prefix image refs with dataset name
            multi_dataset = len(set(img["dataset_id"] for img in top_images)) > 1
            if is_image_query:
                image_context = "\n\n## RELEVANT IMAGES: ##\n"
            else:
                image_context = "\n\nRelevant Images:\n"
            # When building image_context and image_details, prefix with document/slide info if available
            doc_id_to_name = {}
            for img in image_results:
                docid = img.get("document_id")
                if docid and docid not in doc_id_to_name:
                    docid_val = docid
                    # Try to get filename from image metadata
                    if "filename" in img:
                        docid_val = img["filename"]
                    doc_id_to_name[docid] = docid_val
            multi_doc = len(set(img.get("document_id") for img in top_images if img.get("document_id"))) > 1
            for i, img in enumerate(top_images):
                prefix = ""
                if multi_doc and img.get("document_id"):
                    docname = doc_id_to_name.get(img["document_id"], img["document_id"][:8])
                    slide = img.get("slide_number")
                    slide_title = img.get("slide_title")
                    if slide:
                        prefix = f"[{docname}, Slide {slide}] "
                    else:
                        prefix = f"[{docname}] "
                elif multi_dataset:
                    prefix = f"[{dataset_id_to_name.get(img['dataset_id'], img['dataset_id'][:8])}] "
                image_context += f"{prefix}[Image {i+1}] Caption: {img['caption']}\n"
                image_context += f"            This image is available for viewing and download.\n"
            top_image_urls = [img["url"] for img in top_images]
        
        # Combine text and image contexts
        if context_text and image_context:
            # Instead of keeping them separate, integrate image info into the main context
            # This helps the AI be aware of all information sources at once
            image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
            is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
            
            # For image queries, prioritize image information
            if is_image_query and image_results:
                full_context = "## RELEVANT INFORMATION: ##\n\n" + image_context + "\n\n" + context_text
            else:
                full_context = "## RELEVANT INFORMATION: ##\n\n" + context_text + "\n\n" + image_context
        elif context_text:
            full_context = context_text
        elif image_context:
            full_context = image_context
        else:
            full_context = "No relevant information found."
        
        # Get source documents information with indices
        source_documents = []
        seen_documents = set()
        source_images = []
        
        # Add text document sources if available
        if all_contexts:
            for i, (result, metadata) in enumerate(zip(results["documents"][0][:max_contexts], results["metadatas"][0][:max_contexts])):
                if metadata and 'document_id' in metadata and 'source' in metadata:
                    doc_key = f"{metadata['document_id']}_{metadata['source']}"
                    if doc_key not in seen_documents:
                        source_documents.append({
                            'id': metadata['document_id'],
                            'filename': metadata['source'],
                            'dataset_id': dataset_id,
                            'context_index': i + 1  # Store the index used in context_text
                        })
                        seen_documents.add(doc_key)
        
        # Only add images with a decent relevance score (0.2 was the original minimum)
        # Increase threshold to ensure only more relevant images are included
        relevance_threshold = 0.25
        if image_results:
            for i, img in enumerate(top_images):
                if img.get('score', 0) >= relevance_threshold:
                    source_images.append({
                        'id': img['id'],
                        'caption': img['caption'],
                        'url': img['url'],
                        'dataset_id': img['dataset_id'],
                        'type': 'image',
                        'context_index': f"Image {i+1}"
                    })
        
        # Get previous conversation messages to add as context
        conversation_history = ""
        if len(conversation["messages"]) > 1:  # If there's more than just the current message
            # Get last 10 messages maximum
            recent_messages = conversation["messages"][-10:] if len(conversation["messages"]) > 10 else conversation["messages"]
            # Format them for context
            conversation_history = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in recent_messages[:-1]  # Exclude current message
            ])
        
        # Prepare the system instruction and context
        system_instruction = bot.get("system_instruction", "You are a helpful assistant that answers questions based on the provided context.")
        
        # Add specific instructions for handling PowerPoint content
        system_instruction += "\n\nWhen answering questions about PowerPoint presentations, pay particular attention to slide numbers, titles, and slide content structure. Information in slide titles and speaker notes is particularly important. If information appears to be missing or incomplete in the retrieved context, explain that, but still try to provide the best possible answer."
        
        # Updated instructions for better integration of text and images
        system_instruction += "\n\nYou have access to both text documents and images in your knowledge base. Review ALL the provided context carefully, including both text documents AND image captions, before stating that information is not available. Image captions contain valuable information that should be considered as valid sources."
        
        system_instruction += "\n\nWhen referencing images, you MUST follow these exact formatting rules:"
        system_instruction += "\n1. Use the exact format '[Image X]' (where X is the image number) when citing an image"
        system_instruction += "\n2. Place the image citation AFTER describing what the image shows"
        system_instruction += "\n3. Example format: 'Here is an image of a man in a suit holding a martini [Image 1]'"
        system_instruction += "\n4. Never say you can't show or display images - they are automatically displayed when you cite them"
        system_instruction += "\n5. Don't use any other format for image citations (no parentheses, no lowercase 'image', etc.)"
        
        system_instruction += "\n\nWhen a user asks about visual content or specifically mentions images, prioritize relevant images in your response. Only reference images when they are truly relevant to the query. If there are no relevant images for the query, do not mention images at all."
        
        # Add conversation history to system instruction if available
        system_instruction_with_history = system_instruction
        if conversation_history:
            system_instruction_with_history += "\n\nThis is the conversation history so far:\n" + conversation_history
        
        # Check if using model chorus
        if use_model_chorus:
            # Check if a chorus configuration exists
            chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
            os.makedirs(chorus_dir, exist_ok=True)
            
            # Get the user's chorus definitions file
            user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
            
            # Try to load the specific chorus from the user's choruses
            chorus_config = None
            
            if os.path.exists(user_choruses_file) and specific_chorus_id:
                try:
                    with open(user_choruses_file, 'r') as f:
                        choruses = json.load(f)
                        
                    # Find the requested chorus
                    chorus_config = next((c for c in choruses if c["id"] == specific_chorus_id), None)
                    
                    if debug_mode and chorus_config:
                        print(f"Using chorus configuration: {chorus_config.get('name', 'Unnamed')}", flush=True)
                except Exception as e:
                    print(f"Error loading chorus configuration: {str(e)}", flush=True)
            
            # If no chorus config found, return an error
            if not chorus_config:
                # Save the response in the conversation
                bot_response = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "I couldn't find the model chorus configuration. Please check that the chorus exists and is properly configured.",
                    "timestamp": datetime.datetime.now(UTC).isoformat()
                }
                conversation["messages"].append(bot_response)
                with open(conversation_file, 'w') as f:
                    json.dump(conversation, f)
                    
                return jsonify({
                    "response": "I couldn't find the model chorus configuration. Please check that the chorus exists and is properly configured.",
                    "debug": {"error": "Chorus configuration not found"} if debug_mode else None,
                    "conversation_id": conversation_id
                }), 200
            
            # Get response and evaluator models
            response_models = chorus_config.get('response_models', [])
            evaluator_models = chorus_config.get('evaluator_models', [])
            
            # For debugging
            if debug_mode:
                print(f"Using chorus configuration: {chorus_config.get('name', 'Unnamed')}", flush=True)
                print(f"Response models: {len(response_models)}", flush=True)
                print(f"Evaluator models: {len(evaluator_models)}", flush=True)
            
            # Validate the configuration
            if not response_models or not evaluator_models:
                # Save the response in the conversation
                bot_response = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "The model chorus configuration is incomplete. Please configure both response and evaluator models.",
                    "timestamp": datetime.datetime.now(UTC).isoformat()
                }
                conversation["messages"].append(bot_response)
                with open(conversation_file, 'w') as f:
                    json.dump(conversation, f)
                    
                return jsonify({
                    "response": "The model chorus configuration is incomplete. Please configure both response and evaluator models.",
                    "debug": {"error": "Incomplete chorus configuration"} if debug_mode else None,
                    "conversation_id": conversation_id
                }), 200
            
            logs = []
            logs.append(f"Using model chorus: {chorus_config.get('name', 'Unnamed')}")
            
            # Check if diverse RAG is enabled
            use_diverse_rag = chorus_config.get('use_diverse_rag', False)
            if use_diverse_rag:
                logs.append("Diverse RAG mode enabled - each model will receive different contexts")
            
            # Generate responses from all models
            all_responses = []
            anonymized_responses = []
            response_metadata = []
            
            # Process response models
            for model in response_models:
                provider = model.get('provider')
                model_name = model.get('model')
                temperature = float(model.get('temperature', 0.7))
                weight = int(model.get('weight', 1))
                
                # If diverse RAG is enabled, generate a unique context for this model
                model_specific_context = context_text
                if use_diverse_rag:
                    try:
                        # Perform a new search specifically for this model
                        model_contexts = []
                        
                        for dataset_id in dataset_ids:
                            try:
                                # Text-based document retrieval specific to this model
                                collection = chroma_client.get_collection(dataset_id)
                                
                                # Check if collection has any documents
                                collection_count = collection.count()
                                if collection_count > 0:
                                    # Determine how many results to request
                                    n_results = min(5, collection_count)
                                    
                                    # Add slight variation to the query for diversity
                                    model_specific_query = f"{message} relevant to {provider} {model_name}"
                                    
                                    # Get results for this specific model
                                    model_results = collection.query(
                                        query_texts=[model_specific_query],
                                        n_results=n_results
                                    )
                                    
                                    model_contexts.extend(model_results["documents"][0])
                            except Exception as e:
                                logs.append(f"Error with model-specific collection '{dataset_id}': {str(e)}")
                                continue
                        
                        # If we got contexts for this model, use them
                        if model_contexts:
                            # Limit to max_contexts
                            model_contexts = model_contexts[:max_contexts]
                            model_specific_context = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(model_contexts)])
                            logs.append(f"Generated unique context for {provider} {model_name} with {len(model_contexts)} chunks")
                    except Exception as e:
                        logs.append(f"Error generating diverse RAG for {provider} {model_name}: {str(e)}")
                        # Fall back to the common context
                        model_specific_context = context_text
                
                # Create model-specific full context with both text and image information
                model_full_context = model_specific_context
                
                # Add image context to the model prompt if available
                if image_results:
                    # Check if query is likely about images
                    image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration", "logo", "icon"]
                    is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
                    
                    # Format image information - make it more prominent for image queries
                    model_image_context = ""
                    if is_image_query:
                        model_image_context = "\n\n## RELEVANT IMAGES: ##\n"
                    else:
                        model_image_context = "\n\nRelevant Images:\n"
                        
                    # Take top images up to a limit - use more for image queries
                    max_images = 5 if is_image_query else 3
                    top_images = image_results[:max_images]
                    
                    for i, img in enumerate(top_images):
                        model_image_context += f"Available Image {i+1}:\n"
                        model_image_context += f"- Description: {img['caption']}\n"
                        model_image_context += f"- To reference this image in your response, use exactly: [Image {i+1}]\n"
                    
                    # Add image context to the full context
                    if is_image_query:
                        # For image queries, prioritize image information
                        model_full_context = "## RELEVANT INFORMATION: ##\n\n" + model_image_context + "\n\n" + model_specific_context
                    else:
                        model_full_context = "## RELEVANT INFORMATION: ##\n\n" + model_specific_context + "\n\n" + model_image_context
                
                # Add specific instructions for images to the chorus prompt
                image_prompt_instruction = ""
                if image_results:
                    image_prompt_instruction = "\n\nSeveral images were retrieved that may be relevant to this query. If the user's query relates to images or visual information, please reference the images provided in your response using [Image 1], [Image 2], etc. citations. Only reference images when they are directly relevant to answering the question. IMPORTANT: Do NOT state that you cannot display or show images to the user. Images mentioned in the context ARE available to the user for viewing."

                for i in range(weight):
                    try:
                        if provider == 'OpenAI':
                            # Some OpenAI models (like o3) only support default temperature
                            api_temperature = temperature
                            if model_name.startswith('o3-'):
                                api_temperature = 1.0  # o3 models only support default temperature
                            
                            response = openai.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": system_instruction_with_history},
                                    {"role": "user", "content": "Context:\n" + model_full_context + image_prompt_instruction + "\n\nUser question: " + message + "\n\nIf referencing information, please include citations [1], [2], etc. For images, use [Image 1], [Image 2], etc. Only reference images that are directly relevant to answering the question."}
                                ],
                                temperature=api_temperature
                            )
                            response_text = response.choices[0].message.content
                            anonymized_responses.append(response_text)
                            response_metadata.append({
                                "provider": provider,
                                "model": model_name,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                            all_responses.append({
                                "provider": provider,
                                "model": model_name,
                                "response": response_text,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                        elif provider == 'Anthropic':
                            response = anthropic_client.messages.create(
                                model=model_name,
                                system=system_instruction_with_history,
                                messages=[{"role": "user", "content": "Context:\n" + model_full_context + image_prompt_instruction + "\n\nUser question: " + message + "\n\nIf referencing information, please include citations [1], [2], etc. For images, use [Image 1], [Image 2], etc. Only reference images that are directly relevant to answering the question."}],
                                temperature=temperature,
                                max_tokens=1024
                            )
                            response_text = response.content[0].text
                            anonymized_responses.append(response_text)
                            response_metadata.append({
                                "provider": provider,
                                "model": model_name,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                            all_responses.append({
                                "provider": provider,
                                "model": model_name,
                                "response": response_text,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                        elif provider == 'Groq':
                            headers = {
                                "Authorization": "Bearer " + GROQ_API_KEY,
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "model": model_name,
                                "messages": [
                                    {"role": "system", "content": system_instruction_with_history},
                                    {"role": "user", "content": "Context:\n" + model_full_context + image_prompt_instruction + "\n\nUser question: " + message + "\n\nIf referencing information, please include citations [1], [2], etc. For images, use [Image 1], [Image 2], etc. Only reference images that are directly relevant to answering the question."}
                                ],
                                "temperature": temperature,
                                "max_tokens": 1024
                            }
                            response = requests.post(GROQ_API_URL, json=payload, headers=headers)
                            
                            # Check if the request was successful
                            if response.status_code != 200:
                                raise Exception(f"HTTP {response.status_code}: {response.text}")
                            
                            response_json = response.json()
                            
                            # Check if the response has the expected structure
                            if "choices" not in response_json:
                                error_msg = response_json.get("error", {}).get("message", "Unknown error")
                                logs.append(f"Groq API response structure: {list(response_json.keys())}")
                                raise Exception(f"API Error: {error_msg}")
                            
                            if not response_json["choices"] or "message" not in response_json["choices"][0]:
                                logs.append(f"Groq choices structure: {response_json.get('choices', 'N/A')}")
                                raise Exception("Invalid response structure: missing choices or message")
                            
                            response_text = response_json["choices"][0]["message"]["content"]
                            anonymized_responses.append(response_text)
                            response_metadata.append({
                                "provider": provider,
                                "model": model_name,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                            all_responses.append({
                                "provider": provider,
                                "model": model_name,
                                "response": response_text,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                        elif provider == 'Mistral':
                            # Note: This would require a Mistral API implementation
                            # For now we'll use OpenAI as a fallback and log it
                            logs.append(f"Mistral API not implemented, using OpenAI fallback for {model_name}")
                            
                            # Use default temperature for gpt-3.5-turbo (it supports custom temperature, but being safe)
                            fallback_temperature = min(temperature, 2.0)  # gpt-3.5-turbo supports up to 2.0
                            
                            response = openai.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": system_instruction_with_history},
                                    {"role": "user", "content": "Context:\n" + model_full_context + image_prompt_instruction + "\n\nUser question: " + message + "\n\nIf referencing information, please include citations [1], [2], etc. For images, use [Image 1], [Image 2], etc. Only reference images that are directly relevant to answering the question."}
                                ],
                                temperature=fallback_temperature
                            )
                            response_text = response.choices[0].message.content
                            anonymized_responses.append(response_text)
                            response_metadata.append({
                                "provider": "OpenAI (Mistral fallback)",
                                "model": "gpt-3.5-turbo",
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                            all_responses.append({
                                "provider": "OpenAI (Mistral fallback)",
                                "model": "gpt-3.5-turbo",
                                "response": response_text,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                    except Exception as e:
                        logs.append(f"Error with {provider} {model_name}: {str(e)}")
            
            # If no responses, use fallback
            if not all_responses:
                # Save the response in the conversation
                bot_response = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "I encountered an issue with the model chorus. No models were able to generate a response.",
                    "timestamp": datetime.datetime.now(UTC).isoformat()
                }
                conversation["messages"].append(bot_response)
                with open(conversation_file, 'w') as f:
                    json.dump(conversation, f)
                    
                return jsonify({
                    "response": "I encountered an issue with the model chorus. No models were able to generate a response.",
                    "debug": {"error": "No models returned responses", "logs": logs} if debug_mode else None,
                    "conversation_id": conversation_id
                }), 200
            
            # If only one response, return it directly (but still process images)
            if len(all_responses) == 1:
                response_text = all_responses[0]["response"]
                
                # Extract which image indices were referenced in the response using robust regex
                used_image_indices = set()
                if image_results:
                    # Check which images are explicitly referenced in the response
                    print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Checking for image references in response: {len(image_results)} images available", flush=True)
                    
                    # Use robust regex to find all [Image X] references
                    matches = re.findall(r'\[Image (\d+)\]', response_text)
                    for match in matches:
                        idx = int(match)
                        if 1 <= idx <= len(image_results[:5]):
                            used_image_indices.add(idx)
                            print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Response explicitly references [Image {idx}]", flush=True)
                            
                    # If the response mentions images but doesn't use the exact citation format,
                    # include highly relevant images that meet our threshold
                    image_mention_terms = ["image", "picture", "photo", "logo", "diagram", "graph", "visual", "illustration", "icon"]
                    has_image_mentions = any(term.lower() in response_text.lower() for term in image_mention_terms)
                    
                    if has_image_mentions:
                        matching_terms = [term for term in image_mention_terms if term.lower() in response_text.lower()]
                        print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Response contains image-related terms: {matching_terms}", flush=True)
                    
                    # Include relevant images only if the response mentions images or the query is explicitly about images
                    image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
                    is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
                    
                    if (has_image_mentions or is_image_query) and not used_image_indices:
                        # Use a reasonable threshold for image relevance
                        relevance_threshold = 0.3  # Increased from 0.2 for higher quality image matches
                        print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Applying relevance threshold {relevance_threshold} for image queries", flush=True)
                        for i, img in enumerate(image_results[:5]):
                            score = img.get('score', 0)
                            if score >= relevance_threshold:
                                used_image_indices.add(i+1)
                                print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Including Image {i+1} with score {score:.4f} (above threshold {relevance_threshold})", flush=True)
                            else:
                                print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Image {i+1} score {score:.4f} below threshold {relevance_threshold}", flush=True)
                    
                    # Only force-include an image for explicit image queries
                    if is_image_query and not used_image_indices and image_results:
                        # Include only the highest scoring image and only if it has a minimum score
                        min_score_threshold = 0.2
                        best_score = image_results[0].get('score', 0)
                        if best_score >= min_score_threshold:
                            used_image_indices.add(1)  # Add the first (highest scoring) image
                            print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Force including top image with score {best_score:.4f} for image query", flush=True)
                        else:
                            print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Top image score {best_score:.4f} below minimum threshold {min_score_threshold}, not including any images", flush=True)
                    
                    print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Final image selection: {sorted(used_image_indices)}", flush=True)
                
                # Prepare image details for referenced images
                image_details = []
                if used_image_indices and image_results:
                    top_images = image_results[:5]  # Consider up to 5 images
                    multi_dataset = len(set(img["dataset_id"] for img in top_images)) > 1
                    for img_idx in used_image_indices:
                        if 1 <= img_idx <= len(top_images):
                            img = top_images[img_idx-1]
                            base_url = img['url']
                            download_url = f"{base_url}?download=true"
                            prefix = ""
                            if multi_dataset:
                                prefix = f"[{dataset_id_to_name.get(img['dataset_id'], img['dataset_id'][:8])}] "
                            image_details.append({
                                "index": f"{prefix}Image {img_idx}",
                                "caption": img['caption'],
                                "url": img['url'],
                                "download_url": download_url,
                                "id": img['id'],
                                "dataset_id": img['dataset_id'],
                                "document_id": img.get("document_id"),
                                "slide_number": img.get("slide_number"),
                                "slide_title": img.get("slide_title"),
                                "filename": img.get("filename")
                            })
                
                # Debug logging before response
                print(f"CHORUS MODE (SINGLE) - IMAGE DEBUG: image_results count: {len(image_results) if image_results else 0}", flush=True)
                print(f"CHORUS MODE (SINGLE) - IMAGE DEBUG: used_image_indices: {used_image_indices}", flush=True)
                print(f"CHORUS MODE (SINGLE) - IMAGE DEBUG: image_details count: {len(image_details)}", flush=True)
                if image_details:
                    for detail in image_details:
                        print(f"CHORUS MODE (SINGLE) - IMAGE DEBUG: image detail: {detail['index']} -> {detail['url']}", flush=True)
                
                # Save the response in the conversation with image details
                bot_response = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.datetime.now(UTC).isoformat(),
                    "referenced_images": [img["url"] for img in image_details] if image_details else [],
                    "image_details": image_details
                }
                conversation["messages"].append(bot_response)
                with open(conversation_file, 'w') as f:
                    json.dump(conversation, f)
                    
                return jsonify({
                    "response": response_text,
                    "image_details": image_details if image_details else [],  # Always include image_details, even if empty
                    "debug": {
                        "all_responses": all_responses,
                        "anonymized_responses": anonymized_responses,
                        "response_metadata": response_metadata,
                        "logs": logs,
                        "contexts": contexts,
                        "image_results": image_results if image_results else []
                    } if debug_mode else None,
                    "conversation_id": conversation_id
                }), 200
            
            # Get voting from evaluator models
            votes = [0] * len(all_responses)
            
            for model in evaluator_models:
                provider = model.get('provider')
                model_name = model.get('model')
                temperature = float(model.get('temperature', 0.2))
                weight = int(model.get('weight', 1))
                
                voting_prompt = "Here are the " + str(len(anonymized_responses)) + " candidate responses:\n\n" + \
chr(10).join([f"Response {j+1}:\n{resp}" for j, resp in enumerate(anonymized_responses)]) + \
"\n\nWhich response provides the most accurate, helpful, and relevant answer? Return ONLY the number (1-" + \
str(len(anonymized_responses)) + ") of the best response.\n" + \
"Do not reveal any bias or preference based on writing style or approach - evaluate solely on answer quality, accuracy and helpfulness."
                # Apply weight by counting vote multiple times
                for i in range(weight):
                    try:
                        vote_text = ""
                        if provider == 'OpenAI':
                            # Some OpenAI models (like o3) only support default temperature
                            api_temperature = temperature
                            if model_name.startswith('o3-'):
                                api_temperature = 1.0  # o3 models only support default temperature
                            
                            voting_response = openai.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": voting_prompt}],
                                temperature=api_temperature
                            )
                            vote_text = voting_response.choices[0].message.content
                        elif provider == 'Anthropic':
                            voting_response = anthropic_client.messages.create(
                                model=model_name,
                                messages=[{"role": "user", "content": voting_prompt}],
                                temperature=temperature,
                                max_tokens=10
                            )
                            vote_text = voting_response.content[0].text
                        elif provider == 'Groq':
                            headers = {
                                "Authorization": "Bearer " + GROQ_API_KEY,
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "model": model_name,
                                "messages": [{"role": "user", "content": voting_prompt}],
                                "temperature": temperature,
                                "max_tokens": 10
                            }
                            voting_response = requests.post(GROQ_API_URL, json=payload, headers=headers)
                            
                            # Check if the request was successful
                            if voting_response.status_code != 200:
                                raise Exception(f"HTTP {voting_response.status_code}: {voting_response.text}")
                            
                            voting_json = voting_response.json()
                            
                            # Check if the response has the expected structure
                            if "choices" not in voting_json:
                                error_msg = voting_json.get("error", {}).get("message", "Unknown error")
                                logs.append(f"Groq voting API response structure: {list(voting_json.keys())}")
                                raise Exception(f"API Error: {error_msg}")
                            
                            if not voting_json["choices"] or "message" not in voting_json["choices"][0]:
                                logs.append(f"Groq voting choices structure: {voting_json.get('choices', 'N/A')}")
                                raise Exception("Invalid response structure: missing choices or message")
                            
                            vote_text = voting_json["choices"][0]["message"]["content"]
                        elif provider == 'Mistral':
                            # Fallback to OpenAI for Mistral
                            logs.append(f"Mistral API not implemented for evaluation, using OpenAI fallback")
                            
                            # Use safe temperature for gpt-3.5-turbo
                            fallback_temperature = min(temperature, 2.0)  # gpt-3.5-turbo supports up to 2.00
                            
                            voting_response = openai.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": voting_prompt}],
                                temperature=fallback_temperature
                            )
                            vote_text = voting_response.choices[0].message.content
                        
                        # Extract vote number
                        match = re.search(r'\b([1-9][0-9]*)\b', vote_text)
                        if match:
                            vote_number = int(match.group(1))
                            if 1 <= vote_number <= len(anonymized_responses):
                                votes[vote_number-1] += 1
                                logs.append(f"Evaluator {provider} {model_name} voted for response {vote_number}")
                            else:
                                logs.append(f"Invalid vote number: {vote_number}")
                        else:
                            logs.append(f"Could not parse vote from: {vote_text}")
                    except Exception as e:
                        logs.append(f"Error with evaluator {provider} {model_name}: {str(e)}")
            
            # Find winning response
            max_votes = max(votes) if votes else 0
            winning_indices = [i for i, v in enumerate(votes) if v == max_votes]
            winning_index = winning_indices[0] if winning_indices else 0
            
            winning_response = all_responses[winning_index]["response"]
            
            # Extract which image indices were referenced in the winning response using robust regex
            used_image_indices = set()
            if image_results:
                # Check which images are explicitly referenced in the response
                print(f"CHORUS MODE - IMAGE INCLUSION DEBUG: Checking for image references in response: {len(image_results)} images available", flush=True)
                
                # Use robust regex to find all [Image X] references
                matches = re.findall(r'\[Image (\d+)\]', winning_response)
                for match in matches:
                    idx = int(match)
                    if 1 <= idx <= len(image_results[:5]):
                        used_image_indices.add(idx)
                        print(f"CHORUS MODE - IMAGE INCLUSION DEBUG: Response explicitly references [Image {idx}]", flush=True)
                        
                # If the response mentions images but doesn't use the exact citation format,
                # include highly relevant images that meet our threshold
                image_mention_terms = ["image", "picture", "photo", "logo", "diagram", "graph", "visual", "illustration", "icon"]
                has_image_mentions = any(term.lower() in winning_response.lower() for term in image_mention_terms)
                
                if has_image_mentions:
                    matching_terms = [term for term in image_mention_terms if term.lower() in winning_response.lower()]
                    print(f"IMAGE INCLUSION DEBUG: Response contains image-related terms: {matching_terms}", flush=True)
                
                # Include relevant images only if the response mentions images or the query is explicitly about images
                image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
                is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
                
                if (has_image_mentions or is_image_query) and not used_image_indices:
                    # Use a reasonable threshold for image relevance
                    relevance_threshold = 0.3  # Increased from 0.2 for higher quality image matches
                    print(f"IMAGE INCLUSION DEBUG: Applying relevance threshold {relevance_threshold} for image queries", flush=True)
                    for i, img in enumerate(image_results[:5]):
                        score = img.get('score', 0)
                        if score >= relevance_threshold:
                            used_image_indices.add(i+1)
                            print(f"IMAGE INCLUSION DEBUG: Including Image {i+1} with score {score:.4f} (above threshold {relevance_threshold})", flush=True)
                        else:
                            print(f"IMAGE INCLUSION DEBUG: Image {i+1} score {score:.4f} below threshold {relevance_threshold}", flush=True)
                
                # Only force-include an image for explicit image queries
                if is_image_query and not used_image_indices and image_results:
                    # Include only the highest scoring image and only if it has a minimum score
                    min_score_threshold = 0.2
                    best_score = image_results[0].get('score', 0)
                    if best_score >= min_score_threshold:
                        used_image_indices.add(1)  # Add the first (highest scoring) image
                        print(f"IMAGE INCLUSION DEBUG: Force including top image with score {best_score:.4f} for image query", flush=True)
                    else:
                        print(f"IMAGE INCLUSION DEBUG: Top image score {best_score:.4f} below minimum threshold {min_score_threshold}, not including any images", flush=True)
                
                print(f"IMAGE INCLUSION DEBUG: Final image selection: {sorted(used_image_indices)}", flush=True)
            
            # Prepare image details for referenced images
            image_details = []
            if used_image_indices and image_results:
                top_images = image_results[:5]  # Consider up to 5 images
                multi_dataset = len(set(img["dataset_id"] for img in top_images)) > 1
                for img_idx in used_image_indices:
                    if 1 <= img_idx <= len(top_images):
                        img = top_images[img_idx-1]
                        base_url = img['url']
                        download_url = f"{base_url}?download=true"
                        prefix = ""
                        if multi_dataset:
                            prefix = f"[{dataset_id_to_name.get(img['dataset_id'], img['dataset_id'][:8])}] "
                        image_details.append({
                            "index": f"{prefix}Image {img_idx}",
                            "caption": img['caption'],
                            "url": img['url'],
                            "download_url": download_url,
                            "id": img['id'],
                            "dataset_id": img['dataset_id'],
                            "document_id": img.get("document_id"),
                            "slide_number": img.get("slide_number"),
                            "slide_title": img.get("slide_title"),
                            "filename": img.get("filename")
                        })
            
            # Debug logging before response
            print(f"CHORUS MODE - IMAGE DEBUG: image_results count: {len(image_results) if image_results else 0}", flush=True)
            print(f"CHORUS MODE - IMAGE DEBUG: used_image_indices: {used_image_indices}", flush=True)
            print(f"CHORUS MODE - IMAGE DEBUG: image_details count: {len(image_details)}", flush=True)
            if image_details:
                for detail in image_details:
                    print(f"CHORUS MODE - IMAGE DEBUG: image detail: {detail['index']} -> {detail['url']}", flush=True)
            
            # Save the response in the conversation with image details
            bot_response = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": winning_response,
                "timestamp": datetime.datetime.now(UTC).isoformat(),
                "referenced_images": [img["url"] for img in image_details] if image_details else [],
                "image_details": image_details
            }
            conversation["messages"].append(bot_response)
            with open(conversation_file, 'w') as f:
                json.dump(conversation, f)
                
            return jsonify({
                "response": winning_response,
                "image_details": image_details if image_details else [],  # Always include image_details, even if empty
                "debug": {
                    "all_responses": all_responses,
                    "anonymized_responses": anonymized_responses,
                    "response_metadata": response_metadata,
                    "votes": votes,
                    "logs": logs,
                    "contexts": contexts,
                    "image_results": image_results if image_results else []
                } if debug_mode else None,
                "conversation_id": conversation_id
            }), 200
                
    
        # Standard mode - just get a response from OpenAI
        # Enhance system instruction to better handle image queries in mixed datasets
        image_instruction = ""
        if image_results:
            image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
            is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
            
            if is_image_query:
                image_instruction = "\n\nThe user seems to be asking about images. Please prioritize showing and describing relevant images from the provided context when appropriate. When referencing images, use the exact format '[Image X]' (where X is the image number) and be descriptive about what they show."
            else:
                image_instruction = "\n\nRelevant images are available in the context. When they may help answer the user's question, refer to them using the format '[Image X]' (where X is the image number) and briefly describe what they show."
        
        response = openai.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_instruction_with_history + "\n\nWhen you reference information from the provided context, please cite the source using the number in square brackets, e.g. [1], [2], etc. For images, use [Image 1], [Image 2], etc.\n\nIMPORTANT: Before stating that information isn't available, check ALL context including image captions. If information is only found in an image caption, still use that information and cite the image.\n\nIMPORTANT: Do NOT state that you cannot display or show images to the user. When referencing images, simply describe what they contain and cite them with [Image X]. Images mentioned in the context ARE available to the user for viewing."},
                {"role": "user", "content": "Context:\n" + full_context + "\n\nUser question: " + message + "\n\nIf referencing information, please include citations [1], [2], etc. Only reference images that are directly relevant to answering the question."}
            ]
        )
        
        response_text = response.choices[0].message.content
        
        # Extract which context indices were actually used in the response
        used_indices = set()
        for i in range(1, max_contexts + 1):
            if f"[{i}]" in response_text:
                used_indices.add(i)
        
        # Extract which image indices were referenced using robust regex detection
        used_image_indices = set()
        if image_results:
            # Always define top_images here to avoid undefined variable errors
            image_results.sort(key=lambda x: x["score"], reverse=True)
            image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
            is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
            max_images = 5 if is_image_query else 3
            top_images = image_results[:max_images]
            print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Checking for image references in response: {len(image_results)} images available", flush=True)
            
            # Use robust regex to find all [Image X] references
            matches = re.findall(r'\[Image (\d+)\]', response_text)
            for match in matches:
                idx = int(match)
                if 1 <= idx <= len(top_images):
                    used_image_indices.add(idx)
                    print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Response explicitly references [Image {idx}]", flush=True)
                
            # Also check if the image caption is mentioned in the response
            for i in range(1, len(top_images) + 1):
                if top_images[i-1].get('caption'):
                    if top_images[i-1]['caption'].lower() in response_text.lower():
                        used_image_indices.add(i)
                        print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Response mentions image caption for Image {i}", flush=True)
            image_mention_terms = ["image", "picture", "photo", "logo", "diagram", "graph", "visual", "illustration", "icon"]
            has_image_mentions = any(term.lower() in response_text.lower() for term in image_mention_terms)
            if has_image_mentions:
                matching_terms = [term for term in image_mention_terms if term.lower() in response_text.lower()]
                print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Response contains image-related terms: {matching_terms}", flush=True)
            if (has_image_mentions or is_image_query) and not used_image_indices:
                relevance_threshold = 0.3
                print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Applying relevance threshold {relevance_threshold} for image queries", flush=True)
                for i, img in enumerate(top_images):
                    score = img.get('score', 0)
                    if score >= relevance_threshold:
                        used_image_indices.add(i+1)
                        print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Including Image {i+1} with score {score:.4f} (above threshold {relevance_threshold})", flush=True)
                if not used_image_indices and top_images:
                    used_image_indices.add(1)
                    print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Force including top image for image query")
            print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Final image selection: {sorted(used_image_indices)}")
        
        # Filter source_documents to only include those that were actually cited
        used_source_documents = [doc for doc in source_documents if doc['context_index'] in used_indices]
        
        # Filter source_images to only include those that were actually cited
        used_source_images = []
        if image_results:
            for img in source_images:
                img_idx = int(img['context_index'].split()[1])
                if img_idx in used_image_indices:
                    used_source_images.append(img)

        # Create detailed context information for frontend display
        context_details = []
        for i, ctx in enumerate(contexts):
            if i+1 in used_indices:  # Only include contexts that were cited
                # Find the corresponding source document
                source_doc = next((doc for doc in source_documents if doc['context_index'] == i+1), None)
                if source_doc:
                    context_details.append({
                        "index": i+1,
                        "snippet": ctx[:200] + "..." if len(ctx) > 200 else ctx,  # Truncate for display
                        "full_text": ctx,  # Include full snippet text
                        "document_id": source_doc.get('id', ''),
                        "filename": source_doc.get('filename', ''),
                        "dataset_id": source_doc.get('dataset_id', '')
                    })

        # Build image_details directly from top_images and used_image_indices
        image_details = []
        if image_results:
            top_images = image_results[:5]
            multi_dataset = len(set(img["dataset_id"] for img in top_images)) > 1
            for img_idx in used_image_indices:
                if 1 <= img_idx <= len(top_images):
                    img = top_images[img_idx-1]
                    base_url = img['url']
                    download_url = f"{base_url}?download=true"
                    prefix = ""
                    if multi_dataset:
                        prefix = f"[{dataset_id_to_name.get(img['dataset_id'], img['dataset_id'][:8])}] "
                    image_details.append({
                        "index": f"{prefix}Image {img_idx}",
                        "caption": img['caption'],
                        "url": img['url'],
                        "download_url": download_url,
                        "id": img['id'],
                        "dataset_id": img['dataset_id'],
                        "document_id": img.get("document_id"),
                        "slide_number": img.get("slide_number"),
                        "slide_title": img.get("slide_title"),
                        "filename": img.get("filename")
                    })
        
        # Debug logging before response
        print(f"STANDARD MODE - IMAGE DEBUG: image_results count: {len(image_results) if image_results else 0}", flush=True)
        print(f"STANDARD MODE - IMAGE DEBUG: used_image_indices: {used_image_indices}", flush=True)
        print(f"STANDARD MODE - IMAGE DEBUG: image_details count: {len(image_details)}", flush=True)
        if image_details:
            for detail in image_details:
                print(f"STANDARD MODE - IMAGE DEBUG: image detail: {detail['index']} -> {detail['url']}", flush=True)

        # Save the response in the conversation
        bot_response = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.datetime.now(UTC).isoformat(),
            "referenced_images": [img["url"] for img in image_details] if image_details else [],
            "context_details": context_details,  # Add context details to the saved response
            "image_details": image_details  # Add image details to the saved response
        }
        conversation["messages"].append(bot_response)
        with open(conversation_file, 'w') as f:
            json.dump(conversation, f)
            
        # Ensure we have image_details even if no images were found
        if not image_details:
            image_details = []
            
        # Log the final response structure
        print(f"STANDARD MODE - RESPONSE DEBUG: Sending response with {len(image_details)} images", flush=True)
        print(f"STANDARD MODE - RESPONSE DEBUG: Response text contains {response_text.count('[Image')} image references", flush=True)
        
        response_data = {
            "response": response_text,
            "source_documents": used_source_documents,
            "context_details": context_details,  # Include context details in response
            "image_details": image_details,  # Always include image_details array
            "referenced_images": [img["url"] for img in image_details],  # Always include referenced_images array
            "debug": {
                "contexts": contexts,
                "image_results": image_results if image_results else [],
                "used_image_indices": list(used_image_indices) if used_image_indices else []
            } if debug_mode else None,
            "conversation_id": conversation_id
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Error in chat_with_bot: {str(e)}")
        # Save the error response in the conversation
        bot_response = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.",
            "timestamp": datetime.datetime.now(UTC).isoformat()
        }
        conversation["messages"].append(bot_response)
        with open(conversation_file, 'w') as f:
            json.dump(conversation, f)
            
        return jsonify({
            "response": "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.",
            "debug": {"error": str(e)} if debug_mode else None,
            "conversation_id": conversation_id
        }), 200
    
def chat_with_image_handler(user_data, bot_id):
    """Handle chat messages that include an image (multipart or base64)."""
    # 1) multipart vs JSON
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        message        = request.form.get('message', '')
        debug_mode     = request.form.get('debug_mode', 'false').lower() == 'true'
        conversation_id = request.form.get('conversation_id', '')

        if 'image' not in request.files:
            return jsonify({"error": "Image file is required"}), 400
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({"error": "No image selected"}), 400

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        try:
            ext        = os.path.splitext(image_file.filename)[1]
            filename   = f"chat_image_{uuid.uuid4()}{ext}"
            image_path = os.path.join(IMAGE_FOLDER, filename)
            image_file.save(image_path)
            file_ext   = ext.lstrip('.') or 'png'
        except Exception as img_error:
            return jsonify({"error": f"Failed to process image: {img_error}"}), 400

    else:
        data = request.json or {}
        message        = data.get('message', '')
        image_data     = data.get('image_data', '')
        debug_mode     = data.get('debug_mode', False)
        conversation_id = data.get('conversation_id', '')

        if not image_data:
            return jsonify({"error": "Image data is required"}), 400
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        try:
            if ';base64,' in image_data:
                header, encoded = image_data.split(';base64,')
                file_ext = header.split('/')[-1]
            else:
                encoded, file_ext = image_data, 'png'

            decoded = base64.b64decode(encoded)
            filename = f"chat_image_{uuid.uuid4()}.{file_ext}"
            image_path = os.path.join(IMAGE_FOLDER, filename)
            with open(image_path, 'wb') as f:
                f.write(decoded)
        except Exception as img_error:
            return jsonify({"error": f"Failed to process image: {img_error}"}), 400

    # 2) resize
    image_path = resize_image(image_path)

    # 3) default message
    if not message:
        message = "[Image uploaded]"

    # 4) load/create convo
    convo_file = os.path.join(
        CONVERSATIONS_FOLDER,
        f"{user_data['id']}_{bot_id}_{conversation_id}.json"
    )
    if os.path.exists(convo_file):
        with open(convo_file, 'r') as f:
            conversation = json.load(f)
    else:
        conversation = {
            "id": conversation_id,
            "bot_id": bot_id,
            "user_id": user_data['id'],
            "title": (message[:40] + '...') if len(message)>40 else message,
            "created_at": datetime.datetime.now(UTC).isoformat(),
            "updated_at": datetime.datetime.now(UTC).isoformat(),
            "messages": []
        }

    user_msg = {
        "id":        str(uuid.uuid4()),
        "role":      "user",
        "content":   message,
        "timestamp": datetime.datetime.now(UTC).isoformat(),
        "has_image": True,
        "image_path": image_path
    }
    conversation["messages"].append(user_msg)
    conversation["updated_at"] = datetime.datetime.now(UTC).isoformat()
    with open(convo_file, 'w') as f:
        json.dump(conversation, f)

    # 5) call Vision API
    try:
        with open(image_path, 'rb') as img_f:
            b64 = base64.b64encode(img_f.read()).decode('utf-8')

        # load bot system prompt
        bots_dir = os.path.join(os.path.dirname(__file__), "bots")
        with open(os.path.join(bots_dir, f"{user_data['id']}_bots.json")) as bf:
            bots = json.load(bf)
        bot = next((b for b in bots if b["id"] == bot_id), None)
        if not bot:
            return jsonify({"error": "Bot not found"}), 404

        system_inst = bot.get(
            "system_instruction",
            "You are a helpful assistant that can analyze images."
        )
        # include recent history
        recent = conversation["messages"][-10:-1]
        if recent:
            hist = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
            system_inst += "\n\nConversation so far:\n" + hist

        payload = [
            {"role":"system","content": system_inst},
            {"role":"user","content":[
                {"type":"text","text":message if message!="[Image uploaded]" else "What's in this image?"},
                {"type":"image_url","image_url":{"url":f"data:image/{file_ext};base64,{b64}"}}
            ]}
        ]

        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=payload,
            max_tokens=1024
        )
        reply = resp.choices[0].message.content

        bot_msg = {
            "id":        str(uuid.uuid4()),
            "role":      "assistant",
            "content":   reply,
            "timestamp": datetime.datetime.now(UTC).isoformat(),
            "from_image_analysis": True
        }
        conversation["messages"].append(bot_msg)
        with open(convo_file, 'w') as f:
            json.dump(conversation, f)

        image_details = [{
            "index":"Image 1",
            "caption":"Uploaded image",
            "url":f"/api/images/{os.path.basename(image_path)}",
            "download_url":f"/api/images/{os.path.basename(image_path)}?download=true",
            "id": str(uuid.uuid4()),
            "dataset_id": ""
        }]

        return jsonify({
            "response": reply,
            "conversation_id": conversation_id,
            "image_processed": True,
            "image_details": image_details,
            "debug": {
                "image_path": image_path,
                "message": message,
                "history_len": len(recent)
            } if debug_mode else None
        }), 200

    except Exception as e:
        return jsonify({
            "error": "I encountered an error processing your image.",
            "details": str(e) if debug_mode else None
        }), 500