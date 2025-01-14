from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGVector
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisVectorStore
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from utils.logging_util import logger


class VectorStoreHelper:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.logger = logger

    def remove_existing_embeddings(self, source: str, source_type: str, checksum: str):
        """Remove existing document embeddings from vector store

        Args:
            source (str): The identifier of the document source
            source_type (str): The type of the document source
            checksum (str): The checksum of the document, used to uniquely identify the document
        """
        try:
            self.logger.info(f"Removing embeddings for source:{source}, type:{source_type}, checksum:{checksum}")
            
            if isinstance(self.vector_store, RedisVectorStore):
                docs = self.vector_store.search_by_metadata({
                    "source": source,
                    "source_type": source_type,
                    "checksum": checksum
                })
                if docs:
                    self.vector_store.delete([doc.metadata["id"] for doc in docs])

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
                if docs:
                    self.vector_store.delete([doc.metadata["id"] for doc in docs])

            elif isinstance(self.vector_store, QdrantVectorStore):
                # For Qdrant, use the metadata filter directly
                self.logger.info("Using Qdrant's delete method with metadata filter")

                # Create proper Filter object
                filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source)
                        ),
                        FieldCondition(
                            key="metadata.source_type",
                            match=MatchValue(value=source_type)
                        ),
                        FieldCondition(
                            key="metadata.checksum",
                            match=MatchValue(value=checksum)
                        )
                    ]
                )

                # Get points matching the filter using the client directly
                client = self.vector_store.client
                collection_name = self.vector_store.collection_name
                
                # Use scroll to get matching points with correct parameter name
                scroll_result = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter,  # Changed from filter to scroll_filter
                    limit=100  # adjust based on your needs
                )
                
                if scroll_result and scroll_result[0]:
                    # Get point IDs
                    point_ids = [point.id for point in scroll_result[0]]
                    self.logger.info(f"Found {len(point_ids)} matching points")

                    # Delete points by IDs
                    if point_ids:
                        client.delete(
                            collection_name=collection_name,
                            points_selector=point_ids
                        )
                        self.logger.info(f"Deleted {len(point_ids)} points from Qdrant")
                else:
                    self.logger.info("No matching points found to delete")

            else:
                self.logger.warning(f"Unsupported vector store type: {type(self.vector_store)}")

        except Exception as e:
            self.logger.error(f"Error removing embeddings: {str(e)}")
            raise
