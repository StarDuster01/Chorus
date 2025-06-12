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
import faiss
import numpy as np
import os, json
from flask import jsonify, request
from image_processor import image_processor   
# Import the constant from constants.py instead of app.py
from constants import DEFAULT_LLM_MODEL
from constants import IMAGE_FOLDER
from constants import VECTOR_DIMENSION




def remove_image_handler(user_data, dataset_id, image_id):
    """Remove an image from a dataset and rebuild its FAISS index."""
    # 1) validate the dataset belongs to this user
    datasets_dir      = os.path.join(os.path.dirname(__file__), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404

    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)

    # find index
    ds_idx = next((i for i, d in enumerate(datasets) if d["id"] == dataset_id), None)
    if ds_idx is None:
        return jsonify({"error": "Dataset not found"}), 404

    try:
        # locate the image's file path so we can delete it after removal
        image_path = None
        for img in image_processor.image_metadata.get(dataset_id, []):
            if img["id"] == image_id:
                image_path = img.get("path")
                break

        # remove from the in‐memory processor
        success = image_processor.remove_image(dataset_id, image_id)
        if not success:
            return jsonify({"error": "Image not found in dataset"}), 404

        # delete the file on disk
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass

        # rebuild metadata.json + FAISS index
        indices_dir   = os.path.join(os.path.dirname(__file__), "data", "image_indices")
        metadata_file = os.path.join(indices_dir, f"{dataset_id}_metadata.json")
        index_path    = os.path.join(indices_dir, f"{dataset_id}_index.faiss")

        # reload & filter metadata on disk
        if os.path.exists(metadata_file):
            with open(metadata_file,'r') as f:
                meta = json.load(f)
            valid = [m for m in meta if m["id"] != image_id and os.path.exists(m.get("path",""))]
            image_processor.image_metadata[dataset_id] = valid

            # save filtered metadata
            with open(metadata_file,'w') as f:
                json.dump(valid, f)

            # rebuild FAISS index
            idx = faiss.IndexFlatIP(VECTOR_DIMENSION)
            for m in valid:
                emb = m.get("embedding")
                if isinstance(emb, list):
                    idx.add(np.array([emb],dtype=np.float32))
                else:
                    try:
                        e = image_processor.compute_image_embedding(m["path"])
                        idx.add(np.array([e],dtype=np.float32))
                        m["embedding"] = e.tolist()
                    except Exception:
                        pass

            image_processor.image_indices[dataset_id] = idx
            faiss.write_index(idx, index_path)

        # update the dataset's image_count
        actual = len(image_processor.image_metadata.get(dataset_id, []))
        datasets[ds_idx]["image_count"] = actual
        with open(user_datasets_file,'w') as f:
            json.dump(datasets, f)

        return jsonify({"message": "Image removed successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to remove image: {str(e)}"}), 500


def get_dataset_images_handler(user_data, dataset_id):
    """Get all images for a specific dataset"""
    print(f"Getting images for dataset {dataset_id}")
    print(f"Available image metadata keys: {list(image_processor.image_metadata.keys())}")

    # 1) verify dataset file for this user
    datasets_dir      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404

    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)

    # 2) find the dataset index
    dataset_index = next((i for i, ds in enumerate(datasets) if ds["id"] == dataset_id), None)
    if dataset_index is None:
        print(f"Dataset {dataset_id} not found in user datasets")
        return jsonify({"error": "Dataset not found"}), 404

    # prepare for potential index/metadata reload
    indices_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_indices")
    metadata_file = os.path.join(indices_dir, f"{dataset_id}_metadata.json")
    index_path    = os.path.join(indices_dir, f"{dataset_id}_index.faiss")

    # 3) if metadata file exists, validate & rebuild in‐memory structures
    if os.path.exists(metadata_file):
        try:
            print(f"Loading metadata file from {metadata_file}")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            valid = []
            for img_meta in metadata:
                img_meta.setdefault('dataset_id', dataset_id)
                if img_meta['dataset_id']==dataset_id and os.path.exists(img_meta.get('path','')):
                    valid.append(img_meta)
                else:
                    print(f"Skipping {img_meta.get('id')} missing file {img_meta.get('path')}")

            image_processor.image_metadata[dataset_id] = valid

            # rebuild or empty index
            idx = faiss.IndexFlatIP(VECTOR_DIMENSION)
            if os.path.exists(index_path) and valid:
                for m in valid:
                    emb = m.get('embedding')
                    if emb:
                        idx.add(np.array([emb],dtype=np.float32))
                    else:
                        try:
                            e = image_processor.compute_image_embedding(m['path'])
                            idx.add(np.array([e],dtype=np.float32))
                        except Exception as e:
                            print(f"Error embedding {m.get('id')}: {e}")

            image_processor.image_indices[dataset_id] = idx
            image_processor._save_dataset_index(dataset_id)
            print(f"Rebuilt index for {dataset_id} ({len(valid)} images)")

        except Exception as e:
            print(f"Error loading/validating metadata: {e}")

    # 4) now collect the ready metadata for response
    out = []
    for m in image_processor.image_metadata.get(dataset_id, []):
        if m.get('dataset_id')!=dataset_id: continue
        p = m.get('path')
        if not p or not os.path.exists(p): 
            print(f"skip missing {m.get('id')}")
            continue
        c = m.copy()
        c['url'] = f"/api/images/{os.path.basename(p)}"
        c.pop('path',None)
        out.append(c)

    # sort & persist count
    out.sort(key=lambda x: x.get('created_at',''), reverse=True)
    actual = len(out)
    if datasets[dataset_index].get('image_count',0)!=actual:
        datasets[dataset_index]['image_count']=actual
        with open(user_datasets_file,'w') as f:
            json.dump(datasets,f)

    return jsonify({"images": out, "total_images": actual}), 200

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
    
# image_handlers.py  (append near the other handlers)





def search_dataset_images_handler(user_data, dataset_id):
    """
    Search the given dataset for images semantically matching `query`.
    Mirrors original Flask route logic, untouched.
    """
    # ---------- verify dataset ownership ----------
    datasets_dir      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_ds_file      = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")

    if not os.path.exists(user_ds_file):
        return jsonify({"error": "Dataset not found"}), 404

    with open(user_ds_file, "r") as f:
        datasets = json.load(f)

    if not any(ds["id"] == dataset_id for ds in datasets):
        return jsonify({"error": "Dataset not found"}), 404

    # ---------- read request payload ----------
    data  = request.get_json(silent=True) or {}
    query = data.get("query")
    if not query:
        return jsonify({"error": "Search query is required"}), 400
    limit = data.get("limit", 5)

    try:
        # ---------- perform semantic search ----------
        results = image_processor.search_images(dataset_id, query, limit)

        # redact local paths before sending to client
        formatted = []
        for r in results:
            r2 = r.copy()
            if "path" in r2:
                r2["url"] = f"/api/images/{os.path.basename(r2['path'])}"
                del r2["path"]
            formatted.append(r2)

        return jsonify({"query": query, "results": formatted}), 200

    except Exception as e:
        print(f"Error searching images: {e}")
        return jsonify({"error": f"Failed to search images: {e}"}), 500


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
        
        # Generate image with OpenAI
        response = openai.images.generate(**generation_params)
        
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
    

def edit_image_handler(user_data):
    """Edit an image using OpenAI’s image‐edit API"""
    try:
        # 1) multipart/form-data check
        if not request.content_type or not request.content_type.startswith('multipart/form-data'):
            return jsonify({"error": "Request must be multipart/form-data"}), 400

        # 2) get the uploaded image
        if 'image' not in request.files:
            return jsonify({"error": "Image file is required"}), 400
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        # 3) get the prompt
        prompt = request.form.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # 4) optional params
        model         = request.form.get('model', 'gpt-image-1')
        size          = request.form.get('size', '1024x1024')
        quality       = request.form.get('quality', 'medium')
        output_format = request.form.get('output_format', 'png')

        # 5) save source image
        ext      = os.path.splitext(image_file.filename)[1]
        src_name = secure_filename(f"edit_source_{uuid.uuid4()}{ext}")
        src_path = os.path.join(IMAGE_FOLDER, src_name)
        image_file.save(src_path)

        # 6) resize if needed
        src_path = resize_image(src_path)

        # 7) call OpenAI edit API
        response = openai.images.edit(
            image=open(src_path, 'rb'),
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            n=1
        )

        if not getattr(response, 'data', None):
            return jsonify({"error": "No image was generated"}), 500
        image_obj = response.data[0]

        # 8) pull down the edited image bytes
        if getattr(image_obj, 'b64_json', None):
            image_bytes = base64.b64decode(image_obj.b64_json)
        elif getattr(image_obj, 'url', None):
            dl = requests.get(image_obj.url)
            if dl.status_code != 200:
                return jsonify({"error": "Failed to download edited image"}), 500
            image_bytes = dl.content
        else:
            return jsonify({"error": "No image data found in response"}), 500

        # 9) save edited image
        out_name = f"edited_{uuid.uuid4()}.{output_format}"
        out_path = os.path.join(IMAGE_FOLDER, out_name)
        with open(out_path, 'wb') as f:
            f.write(image_bytes)

        api_url = f"/api/images/{out_name}"
        return jsonify({
            "success":   True,
            "image_url": api_url,
            "images":    [{"image_url": api_url}],
            "params": {
                "model":  model,
                "size":   size,
                "quality": quality,
                "format":  output_format
            }
        }), 200

    except Exception as e:
        print(f"Error editing image: {e}")
        return jsonify({"error": f"Image editing failed: {e}"}), 500