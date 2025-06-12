"""
Handlers for document-level operations (download, content, context, etc.).
All functions **return** the same tuple that a Flask route expects:
    (jsonify({...}), status_code)
They contain no route decorators – those stay in app.py.
"""

import os, json
from flask import jsonify, send_file
from text_extractors import extract_text_from_file       # already in your project
import chroma_client                                      # your wrapper
from datetime import UTC                                  # only for logging – optional

from constants import DOCUMENT_FOLDER

# document_handlers.py   (add this just under get_document_content_handler)

from flask import jsonify, request
import os, json
import chroma_client                     # existing import in your project

# document_handlers.py  (append below previous handlers)

from flask import jsonify
import os, json
import chroma_client
from text_extractors import extract_text_from_file


def download_document_handler(user_data, dataset_id, document_id, filename):
    """Download a previously‐uploaded document file."""
    # 1) verify the dataset file exists for this user
    datasets_dir      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404

    # 2) load and check membership
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
    if not any(ds["id"] == dataset_id for ds in datasets):
        return jsonify({"error": "Dataset not found"}), 404

    # 3) locate the actual file
    document_path = None
    # direct match
    direct = os.path.join(DOCUMENT_FOLDER, f"{document_id}_{filename}")
    if os.path.exists(direct):
        document_path = direct
    else:
        # fallback: any file that ends with _filename
        for fname in os.listdir(DOCUMENT_FOLDER):
            if fname.endswith(f"_{filename}"):
                document_path = os.path.join(DOCUMENT_FOLDER, fname)
                break

    # 4) not found?
    if not document_path or not os.path.exists(document_path):
        return jsonify({"error": "Document not found"}), 404

    # 5) send it
    return send_file(document_path, as_attachment=True, download_name=filename)

def get_original_document_handler(user_data, document_id):
    """
    Return the full text (and basic metadata) of the *source* file
    that was chunk-indexed in ChromaDB, if that file is still on disk.
    """
    try:
        # -------- locate user datasets --------
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        user_ds_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")

        if not os.path.exists(user_ds_file):
            return jsonify({"error": "No datasets found"}), 404

        with open(user_ds_file, "r") as f:
            datasets = json.load(f)

        # -------- pull ONE chunk to grab metadata --------
        doc_md = None
        for ds in datasets:
            ds_id = ds["id"]
            try:
                col  = chroma_client.get_collection(ds_id)
                res  = col.get(where={"document_id": document_id}, limit=1)
                if res and res["metadatas"]:
                    doc_md = res["metadatas"][0]
                    break
            except Exception as e:
                print(f"get_original_document_handler – query {ds_id} failed: {e}")

        if not doc_md:
            return jsonify({"error": "Document not found"}), 404

        file_path = doc_md.get("file_path", "")
        filename  = doc_md.get("filename", "Unknown document")

        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "Original file not found"}), 404

        # -------- read original file --------
        try:
            content = extract_text_from_file(file_path)
        except Exception as e:
            return jsonify({"error": f"Error extracting text from file: {e}"}), 500

        ext = os.path.splitext(file_path)[1].lower().lstrip(".")

        return jsonify({
            "document_id": document_id,
            "filename"   : filename,
            "content"    : content,
            "file_type"  : ext,
        }), 200

    except Exception as e:
        print(f"Unhandled error in get_original_document_handler: {e}")
        return jsonify({"error": f"Failed to retrieve original document: {e}"}), 500


def get_context_snippet_handler(user_data, document_id):
    """
    Return one chunk (or the whole set of chunks) for a document,
    plus metadata so the frontend can decide whether to show the “view original” link.
    """
    try:
        # ------------- locate all datasets for this user -------------
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")

        if not os.path.exists(user_datasets_file):
            return jsonify({"error": "No datasets found"}), 404

        with open(user_datasets_file, "r") as f:
            datasets = json.load(f)

        # ------------- pull desired chunk(s) -------------
        chunk_index        = request.args.get("chunk")       # may be None
        document_chunks    = []
        document_metadata  = None

        for ds in datasets:
            ds_id = ds["id"]
            try:
                col = chroma_client.get_collection(ds_id)

                if chunk_index is not None:
                    chunk_id = f"{document_id}_{chunk_index}"
                    res = col.get(ids=[chunk_id])
                else:
                    res = col.get(where={"document_id": document_id})

                if res and res["documents"]:
                    document_chunks.extend(zip(res["documents"], res["metadatas"]))
                    if not document_metadata and res["metadatas"]:
                        document_metadata = res["metadatas"][0]
            except Exception as e:
                print(f"get_context_snippet_handler – error querying {ds_id}: {e}")

        if not document_chunks:
            return jsonify({"error": "Document chunk not found"}), 404

        # order by chunk idx
        document_chunks.sort(key=lambda x: x[1].get("chunk", 0) if x[1] else 0)

        if chunk_index is not None:
            snippet_content = document_chunks[0][0]
        else:
            snippet_content = "\n\n".join(chunk[0] for chunk in document_chunks)

        # ---------- metadata ----------
        md        = document_metadata or {}
        filename  = md.get("filename", "Unknown document")
        file_path = md.get("file_path", "")
        source    = md.get("source", filename)
        original_file_exists = bool(file_path and os.path.exists(file_path))

        return jsonify({
            "document_id"          : document_id,
            "filename"             : filename,
            "source"               : source,
            "content"              : snippet_content,
            "original_file_exists" : original_file_exists,
            "chunk_index"          : chunk_index,
        }), 200

    except Exception as e:
        print(f"Unhandled error in get_context_snippet_handler: {e}")
        return jsonify({"error": f"Failed to retrieve context snippet: {e}"}), 500



def get_document_content_handler(user_data, document_id):
    """Return the full text of a document reconstructed from ChromaDB chunks."""
    try:
        # ---------- locate the user’s datasets ----------
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")

        if not os.path.exists(user_datasets_file):
            return jsonify({"error": "No datasets found"}), 404

        with open(user_datasets_file, "r") as f:
            datasets = json.load(f)

        # ---------- pull every chunk for that document ----------
        document_chunks, document_metadata = [], None
        for dataset in datasets:
            dataset_id = dataset["id"]
            try:
                collection = chroma_client.get_collection(dataset_id)
                res = collection.get(where={"document_id": document_id})

                if res and res["documents"]:
                    document_chunks.extend(zip(res["documents"], res["metadatas"]))
                    if not document_metadata and res["metadatas"]:
                        document_metadata = res["metadatas"][0]
            except Exception as e:
                print(f"[{UTC.now().isoformat()}] get_document_content_handler:"
                      f" error querying collection {dataset_id}: {e}")
                continue

        if not document_chunks:
            return jsonify({"error": "Document not found"}), 404

        # ---------- rebuild & return ----------
        document_chunks.sort(key=lambda x: x[1].get("chunk", 0) if x[1] else 0)
        full_content = "\n\n".join(chunk[0] for chunk in document_chunks)

        filename   = (document_metadata or {}).get("filename", "Unknown document")
        file_path  = (document_metadata or {}).get("file_path", "")
        original_content = full_content

        if file_path and os.path.exists(file_path):
            try:
                original_content = extract_text_from_file(file_path) or full_content
            except Exception as e:
                print(f"Error extracting original file: {e}")

        return jsonify({
            "document_id": document_id,
            "filename": filename,
            "content": original_content,
        }), 200

    except Exception as e:
        print(f"Unhandled error in get_document_content_handler: {e}")
        return jsonify({"error": f"Failed to retrieve document content: {e}"}), 500
    
