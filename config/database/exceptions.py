class DatabaseError(Exception):
    """Base class for database exceptions"""
    pass

class EntityNotFoundError(DatabaseError):
    """Raised when an entity is not found in the database"""
    def __init__(self, entity_type: str, entity_id: any):
        self.message = f"{entity_type} with id {entity_id} not found"
        super().__init__(self.message)

class DuplicateEntityError(DatabaseError):
    """Raised when attempting to create a duplicate entity"""
    def __init__(self, entity_type: str, identifier: str):
        self.message = f"Duplicate {entity_type} with identifier {identifier}"
        super().__init__(self.message)

class DatabaseConnectionError(DatabaseError):
    """Raised when there's an issue connecting to the database"""
    def __init__(self, original_error: Exception):
        self.message = f"Database connection error: {str(original_error)}"
        super().__init__(self.message) 