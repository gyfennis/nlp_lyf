import os


def rewrite_image_paths_in_chunk(text: str, file_path: str, processed_root: str = "processed") -> str:
    """Replace relative `images/` paths with processed directory paths for serving."""
    file_dir = os.path.basename(file_path).split(".")[0]
    prefix = os.path.join(".", processed_root, file_dir, "vlm", "images")
    return text.replace("images/", prefix + "/")
