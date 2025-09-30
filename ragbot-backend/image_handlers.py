import os
import base64
import uuid
import datetime
from datetime import UTC
import io
import requests
from werkzeug.utils import secure_filename
from flask import send_from_directory, send_file, jsonify, request
from PIL import Image
import openai
# Import the constant from constants.py instead of app.py
from constants import DEFAULT_LLM_MODEL

# Helper function to resize large images
def resize_image(image_path, max_dimension=1024):
    """Resize an image if it's larger than max_dimension in either dimension
    
    Args:
        image_path: Path to the image file
        max_dimension: Maximum size for either dimension
        
    Returns:
        str: Path to the resized image (same as input if no resize needed)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Check if resize needed
            if width <= max_dimension and height <= max_dimension:
                return image_path
                
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
                
            # Resize the image
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the resized image (overwrite original)
            resized.save(image_path, optimize=True, quality=85)
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            return image_path
    except Exception as e:
        print(f"Error resizing image: {str(e)}")
        return image_path  # Return original path if resize fails

def generate_image_handler(user_data, image_folder):
    """Handle image generation requests"""
    try:
        # Get request data
        data = request.json
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
            
        # Get additional parameters with defaults for gpt-image-1
        model = "gpt-image-1"  # Always use gpt-image-1
        size = data.get('size', '1024x1024')  # Default square
        quality = data.get('quality', 'auto')  # 'low', 'medium', 'high', or 'auto'
        n = data.get('n', 1)  # Number of images to generate
        output_format = data.get('output_format', 'png')  # 'png', 'jpeg', or 'webp'
        moderation = data.get('moderation', 'auto')  # 'auto' or 'low'
        
        # Build the API request params with only supported parameters
        generation_params = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality,
            "moderation": moderation
        }
        
        # Set dedicated API key for image generation
        import os
        original_api_key = openai.api_key
        image_gen_api_key = os.getenv("OPENAI_IMAGE_API_KEY")
        if not image_gen_api_key:
            print("WARNING: OPENAI_IMAGE_API_KEY not set, using default OPENAI_API_KEY")
            image_gen_api_key = openai.api_key
        openai.api_key = image_gen_api_key
        
        try:
            # Generate image with OpenAI
            response = openai.images.generate(**generation_params)
        finally:
            # Always restore the original API key
            openai.api_key = original_api_key
        
        results = []
        # Process all generated images
        for i, image_obj in enumerate(response.data):
            # Get the base64 content from OpenAI response
            if hasattr(image_obj, 'b64_json') and image_obj.b64_json:
                # Decode base64 content
                image_content = base64.b64decode(image_obj.b64_json)
                image_url = None
            elif hasattr(image_obj, 'url') and image_obj.url:
                # For backward compatibility if URL is provided
                image_url = image_obj.url
                
                # Download the image
                image_response = requests.get(image_url)
                if image_response.status_code != 200:
                    return jsonify({"error": f"Failed to download generated image {i+1}"}), 500
                    
                image_content = image_response.content
            else:
                return jsonify({"error": f"No image data found in response for image {i+1}"}), 500
                
            # Create a unique filename
            ext = output_format if output_format else "png"  # Default to png
            filename = f"generated_{str(uuid.uuid4())}.{ext}"
            filepath = os.path.join(image_folder, filename)
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(image_content)
            
            # Create the URL for accessing the image
            api_image_url = f"/api/images/{filename}"
            
            # Add to results
            results.append({
                "image_url": api_image_url,
                "original_url": image_url,
                "params": {
                    "model": model,
                    "size": size,
                    "quality": quality,
                    "format": output_format
                }
            })
        
        return jsonify({
            "success": True,
            "images": results
        })
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": f"Image generation failed: {str(e)}"}), 500

def enhance_prompt_handler(user_data):
    """Handle prompt enhancement for image generation"""
    try:
        print("\n=== Starting Prompt Enhancement ===")
        # Get request data
        data = request.json
        original_prompt = data.get('prompt')
        print(f"Received prompt: {original_prompt}")
        
        if not original_prompt:
            print("Error: No prompt provided")
            return jsonify({"error": "Prompt is required"}), 400
        
        # Craft system message for the LLM to enhance the prompt
        system_message = """You are an expert at creating detailed image prompts for AI image generation. 
Your task is to enhance the user's prompt by adding more descriptive elements, artistic style, lighting, mood, and details.
Keep the original intent and subject matter, but make it much more detailed and visually compelling.
Respond with ONLY the enhanced prompt text, nothing else. No explanations or additional text."""

        print(f"Calling {DEFAULT_LLM_MODEL} for enhancement...")
        # Call the LLM to enhance the prompt
        response = openai.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Enhance this image prompt: {original_prompt}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        enhanced_prompt = response.choices[0].message.content.strip()
        print(f"Enhanced prompt: {enhanced_prompt}")
        
        result = {
            "success": True,
            "original_prompt": original_prompt,
            "enhanced_prompt": enhanced_prompt
        }
        print("=== Prompt Enhancement Complete ===\n")
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error enhancing prompt: {str(e)}")
        print(f"Full error details: {e}")
        print("=== Prompt Enhancement Failed ===\n")
        return jsonify({"error": f"Prompt enhancement failed: {str(e)}"}), 500

def get_image_handler(image_folder, filename):
    """Serve an uploaded image file"""
    # Check if client accepts image formats
    accept_header = request.headers.get('Accept', '')
    if accept_header and '*/*' not in accept_header and 'image/' not in accept_header:
        return jsonify({"error": "Client doesn't accept image format"}), 406
    
    # Check if download parameter is provided
    download = request.args.get('download', 'false').lower() == 'true'
    
    if download:
        # If download is requested, set Content-Disposition header
        return send_from_directory(
            image_folder, 
            filename, 
            as_attachment=True,
            download_name=filename
        )
    else:
        # Regular image view
        return send_from_directory(image_folder, filename) 