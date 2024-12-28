from pathlib import Path
import os



base_dir = os.getenv('BASE_DIR')

if base_dir is None:
    raise ValueError("BASE_DIR environment variable is not set")

BASE_DIR = Path(base_dir)



