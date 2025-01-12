from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGVector
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisVectorStore


class VectorStoreHelper:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def remove_existing_embeddings(self, source: str, source_type: str, checksum: str):
        """Remove existing document embeddings from vector store

        Args:
            source (str): The identifier of the document source
            source_type (str): The type of the document source
            checksum (str): The checksum of the document, used to uniquely identify the document
        """
        docs = []  # Initialize docs before the conditional blocks
        
        # Search for document embeddings that match the source and checksum in the vector store
        if isinstance(self.vector_store, RedisVectorStore):
            docs = self.vector_store.search_by_metadata({
                "source": source,
                "source_type": source_type,
                "checksum": checksum
            })
        elif isinstance(self.vector_store, PGVector):
            # Create a filter condition for metadata
            filter = {
                "metadata": {
                    "source": source,
                    "source_type": source_type,
                    "checksum": checksum
                }
            }
            # Use a dummy vector since we only care about metadata
            dummy_vector = [0.0] * self.embeddings.dimension
            docs = self.vector_store.similarity_search_by_vector(
                embedding=dummy_vector,
                k=100,  # Adjust based on expected max documents
                filter=filter
            )
        elif isinstance(self.vector_store, QdrantVectorStore):
            # For Qdrant, use the metadata filter directly
            filter = {
                "must": [
                    {
                        "key": "metadata.source",
                        "match": {"value": source}
                    },
                    {
                        "key": "metadata.source_type",
                        "match": {"value": source_type}
                    },
                    {
                        "key": "metadata.checksum",
                        "match": {"value": checksum}
                    }
                ]
            }
            # Qdrant's delete method with proper metadata filtering
            self.vector_store.delete(filter=filter)
            return  # Early return as Qdrant handles deletion directly
        else:
            self.logger.warning(f"Unsupported vector store type: {type(self.vector_store)}")

        # If matching document embeddings are found, delete them from the vector store
        if docs:
            self.vector_store.delete([doc.metadata["id"] for doc in docs])
