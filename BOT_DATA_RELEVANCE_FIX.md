# Bot Data Relevance Fix - Include Images as Relevant Data

## Problem Identified
Bots were showing "I don't have any documents or images in my knowledge base yet" even when they had relevant images available, because the system only considered images as "relevant data" for explicit image queries.

## Root Cause
The original logic in `chat_handler.py` had this flow:
1. Check if user query contains image-related terms
2. If NOT an image query → skip all image processing
3. Only populate `image_results` for explicit image queries
4. Check `if not all_contexts and not image_results` → show "no data" message

This meant that for non-image queries, `image_results` was always empty, so bots with only images (or mostly images) would appear to have no data.

## Solution Implemented

### 1. Modified Image Query Detection Logic
**Before:**
```python
# Skip all image processing if this is not an image query
if not is_image_query:
    print(f"IMAGE SEARCH DEBUG: Skipping image retrieval for non-image query: '{message}'", flush=True)
    continue
```

**After:**
```python
# Check dataset type and image count from metadata first
# ... check if dataset has images ...

# Skip image retrieval if this is not an image query AND dataset has no images
# But always mark dataset as having data if images exist, even for non-image queries
if not is_image_query and not (has_images or metadata_count > 0):
    print(f"IMAGE SEARCH DEBUG: Skipping image retrieval for non-image query with no images: '{message}'", flush=True)
    continue
```

### 2. Added Representative Image Retrieval for Non-Image Queries
**New Logic:**
- For **image queries**: Search with user's specific query (as before)
- For **non-image queries**: Get representative images using generic search
- Always populate `image_results` when dataset has images

**Implementation:**
```python
if is_image_query:
    # For image queries, search with the user's specific query
    img_results = image_processor.search_images(dataset_id, message, top_k=top_k)
else:
    # For non-image queries, just get some representative images to show the bot has data
    generic_query = "image"
    img_results = image_processor.search_images(dataset_id, generic_query, top_k=top_k)
```

### 3. Adjusted Search Parameters
- **Image queries**: 6-8 images (unchanged)
- **Non-image queries**: 1-2 images (just enough to show bot has data)

## Result
Now when a bot has relevant images in its datasets:

✅ **Before fix**: 
- Image query: "Show me images" → Works, finds images
- Non-image query: "Tell me about X" → "I don't have any documents or images..."

✅ **After fix**:
- Image query: "Show me images" → Works, finds relevant images  
- Non-image query: "Tell me about X" → Proceeds with available text + representative images

## Impact
- Bots with image-only or image-heavy datasets will no longer appear to have "no data"
- Non-image queries still work normally but bot knows it has data available
- Image queries continue to work as before
- Better user experience - no false "empty knowledge base" messages

## Technical Details
- Modified: `ragbot-backend/handlers/chat_handler.py`
- Lines affected: ~304-340 (image processing logic)
- Backward compatible: No breaking changes to existing functionality
- Performance impact: Minimal (only 1-2 extra image searches for non-image queries) 