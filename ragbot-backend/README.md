# RagBot Backend

## Image Generation Functionality

The backend now supports generating and editing images using OpenAI's GPT-Image-1 model.

### Generating Images

You can generate images using the `/api/images/generate` endpoint:

```
POST /api/images/generate
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

{
  "prompt": "A children's book drawing of a veterinarian using a stethoscope to listen to the heartbeat of a baby otter",
  "model": "gpt-image-1",
  "size": "1024x1024",
  "quality": "high", 
  "output_format": "png",
  "background": "transparent",
  "output_compression": 80,
  "moderation": "low"
}
```

**Parameters:**

- `prompt` (required): Text description of the image you want to generate
- `model`: One of: "gpt-image-1" (recommended), "dall-e-3", or "dall-e-2"
- `size`: Image dimensions, options vary by model:
  - GPT-Image-1: "1024x1024", "1536x1024", "1024x1536", or "auto"
  - DALL-E 3: "1024x1024", "1792x1024", or "1024x1792"
  - DALL-E 2: "256x256", "512x512", or "1024x1024"
- `quality`: 
  - GPT-Image-1: "low", "medium", "high", or "auto"
  - DALL-E 3: "standard" or "hd"
- `output_format`: "png", "jpeg", or "webp" (GPT-Image-1 only, used for file extension)
- `background`: "transparent" or "auto" (GPT-Image-1 only, works with png and webp)
- `output_compression`: 0-100 compression level for jpeg or webp (GPT-Image-1 only)
- `moderation`: "auto" or "low" (GPT-Image-1 only)

**Note:** The GPT-Image-1 API differs from the DALL-E API:
- GPT-Image-1 always returns base64-encoded images
- The `output_format` parameter is only used to set the file extension when saving the image locally
- Parameters like `style` and `response_format` are not supported by GPT-Image-1
- DALL-E 3 supports both URL and base64 response formats, while GPT-Image-1 only supports base64

### Editing Images

You can edit images using the `/api/images/edit` endpoint:

```
POST /api/images/edit
Content-Type: multipart/form-data
Authorization: Bearer YOUR_TOKEN

prompt: "Generate a photorealistic image of a gift basket containing all the items in the reference pictures."
model: "gpt-image-1"
quality: "high"
size: "1024x1024"
output_format: "png"
image: [file1, file2, file3, ...] # Multiple image files can be uploaded
mask: [mask_file] # Optional mask file
```

**Parameters:**

- `prompt` (required): Text description of the desired edits
- `model`: "gpt-image-1" (recommended) or "dall-e-2"
- `image`: One or more image files (GPT-Image-1 supports multiple reference images)
- `mask`: Optional mask file where transparent areas indicate where to edit
- Additional parameters as in image generation

### Image Sizes and Quality

For `gpt-image-1`, different quality settings use different token counts:

| Quality | Square (1024×1024) | Portrait (1024×1536) | Landscape (1536×1024) |
|---------|-------------------|---------------------|----------------------|
| Low     | 272 tokens        | 408 tokens          | 400 tokens           |
| Medium  | 1056 tokens       | 1584 tokens         | 1568 tokens          |
| High    | 4160 tokens       | 6240 tokens         | 6208 tokens          |

### Response

Both endpoints return a JSON response with:

```json
{
  "image_url": "/api/images/generated-image-filename.png",
  "filename": "generated-image-filename.png"
}
```

You can access the image directly at the provided URL. 