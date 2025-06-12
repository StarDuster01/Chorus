def convert_wmf_to_png(wmf_path):
    """Convert WMF file to PNG format
    
    Args:
        wmf_path: Path to WMF file
        
    Returns:
        str: Path to converted PNG file
    """
    try:
        from wand.image import Image as WandImage
        
        # Create PNG path
        png_path = os.path.splitext(wmf_path)[0] + '.png'
        
        # Convert WMF to PNG using ImageMagick
        with WandImage(filename=wmf_path) as img:
            img.format = 'png'
            img.save(filename=png_path)
            
        return png_path
    except ImportError:
        print("Warning: Wand (ImageMagick) not available. Cannot convert WMF files.")
        return None
    except Exception as e:
        print(f"Error converting WMF to PNG: {str(e)}")
        return None