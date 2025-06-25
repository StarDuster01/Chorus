import os
DEFAULT_LLM_MODEL = "o3-2025-04-16"
IMAGE_GENERATION_MODEL = "gpt-image-1"  
VECTOR_DIMENSION = 512
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Root external storage folder (sibling to the code repo)
STORAGE_DIR = os.environ.get("STORAGE_DIR", "/ChorusAllData")
DATASETS_FOLDER = os.path.join(STORAGE_DIR, "datasets")
CONVERSATIONS_FOLDER = os.path.join(STORAGE_DIR, "conversations")
UPLOAD_FOLDER = os.path.join(STORAGE_DIR, "uploads")  # override old uploads path
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
DOCUMENT_FOLDER = os.path.join(UPLOAD_FOLDER, "documents")
IMAGE_INDICES = os.path.join(STORAGE_DIR, "image_indices")
USERS_FOLDER = os.path.join(STORAGE_DIR, "users")
BOTS_FOLDER = os.path.join(STORAGE_DIR, "bots")
CHORUSES_FOLDER = os.path.join(STORAGE_DIR, "choruses")
text_extensions = ['.pdf', '.docx', '.txt', '.pptx']
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']

# Ensure all external storage directories exist at startup
for d in (DATASETS_FOLDER, CONVERSATIONS_FOLDER, UPLOAD_FOLDER,
          IMAGE_FOLDER, DOCUMENT_FOLDER, IMAGE_INDICES, USERS_FOLDER, BOTS_FOLDER, CHORUSES_FOLDER):
    os.makedirs(d, exist_ok=True)