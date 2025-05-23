#!/usr/bin/env python3
"""
Utility script to fix image paths for images that were uploaded before the path fix.
This moves images from uploads/ to uploads/images/ if they're image files.
"""

import os
import shutil
import argparse

def fix_image_paths(dry_run=True):
    """Move image files from uploads/ to uploads/images/"""
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uploads_dir = os.path.join(script_dir, "uploads")
    images_dir = os.path.join(uploads_dir, "images")
    
    # Ensure images directory exists
    os.makedirs(images_dir, exist_ok=True)
    
    # Define image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
    
    if not os.path.exists(uploads_dir):
        print(f"Uploads directory doesn't exist: {uploads_dir}")
        return
    
    # Find image files in uploads/ that should be in uploads/images/
    moved_count = 0
    
    for filename in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir, filename)
        
        # Skip if it's a directory
        if os.path.isdir(file_path):
            continue
            
        # Check if it's an image file by extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in image_extensions:
            source_path = file_path
            dest_path = os.path.join(images_dir, filename)
            
            # Check if destination already exists
            if os.path.exists(dest_path):
                print(f"Skipping {filename} - already exists in images folder")
                continue
            
            if dry_run:
                print(f"Would move: {source_path} -> {dest_path}")
            else:
                try:
                    shutil.move(source_path, dest_path)
                    print(f"Moved: {filename} to images folder")
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {filename}: {str(e)}")
    
    if dry_run:
        print(f"\nDry run complete. {moved_count} images would be moved.")
        print("Run with --execute to actually move the files.")
    else:
        print(f"\nMoved {moved_count} images to the correct folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix image paths by moving images to correct folder")
    parser.add_argument("--execute", action="store_true", help="Actually move the files (default is dry run)")
    
    args = parser.parse_args()
    
    if args.execute:
        print("Moving images to correct folder...")
        fix_image_paths(dry_run=False)
    else:
        print("Dry run - showing what would be moved...")
        fix_image_paths(dry_run=True) 