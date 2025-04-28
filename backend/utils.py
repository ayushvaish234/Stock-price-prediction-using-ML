import os

def delete_existing_file(file_path):
    """Delete the file if it already exists."""
    if os.path.exists(file_path):
        os.remove(file_path)


def ensure_directory_exists(directory):
    """Ensure that the specified directory exists. Create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)