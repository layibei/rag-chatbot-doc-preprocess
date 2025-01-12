from ulid import ULID

def get_id() -> str:
    """Generate a unique ID using ULID"""
    return str(ULID())  # Convert ULID to string directly
