import os
import traceback
from typing import List, Dict, Any

from langchain_core.documents import Document
from neo4j import GraphDatabase

from config.common_settings import CommonConfig
from utils.logging_util import logger

# Get the absolute path of the current file
CURRENT_FILE_PATH = os.path.abspath(__file__)
# Get the directory containing the current file
BASE_DIR = os.path.dirname(CURRENT_FILE_PATH)


class GraphStoreHelper:
    def __init__(self, graph_db: GraphDatabase, config: CommonConfig):
        self.logger = logger
        self.driver = graph_db
        self.config = config

        if not self.driver:
            self.logger.warning("No Neo4j driver provided - graph store operations will be disabled")
            return
        self.nlp = self.config.get_nlp_spacy()

    def find_related_chunks(self, query: str, k: int = 3) -> List[Document]:
        """Find related chunks using graph traversal with versioning support"""
        try:
            entities = self._extract_entities(query)

            with self.driver.session() as session:
                # Enhanced query to consider only the latest document versions
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name IN $entity_names
                    MATCH (e)<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
                    WHERE NOT EXISTS((d)<-[:REPLACES]-())  // Only latest versions
                    WITH c, d, count(DISTINCT e) as relevance,
                         collect(DISTINCT e.name) as matched_entities
                    ORDER BY relevance DESC
                    LIMIT $k
                    RETURN c.content as content,
                           d.doc_id as doc_id,
                           d.source as source,
                           d.source_type as source_type,
                           d.checksum as checksum,
                           matched_entities,
                           relevance
                """,
                                     entity_names=[e["text"] for e in entities],
                                     k=k
                                     )

                # Convert to LangChain Documents
                return [
                    Document(
                        page_content=record["content"],
                        metadata={
                            "doc_id": record["doc_id"],
                            "source": record["source"],
                            "source_type": record["source_type"],
                            "checksum": record["checksum"],
                            "matched_entities": record["matched_entities"],
                            "graph_relevance_score": record["relevance"],
                            "retrieval_type": "graph"
                        }
                    ) for record in result
                ]

        except Exception as e:
            self.logger.error(f"Error finding related chunks: {str(e)}, stack: {traceback.format_exc()}")
            return []

    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract entities using spaCy"""
        try:
            doc = self.nlp(text)
            entities = [{"text": ent.text, "label": ent.label_}
                        for ent in doc.ents]

            # Add noun phrases as potential entities
            noun_phrases = [{"text": np.text, "label": "NOUN_PHRASE"}
                            for np in doc.noun_chunks
                            if len(np.text.split()) > 1]  # Only multi-word phrases

            return entities + noun_phrases

        except Exception as e:
            self.logger.error(f"Error in entity extraction: {str(e)}")
            return []

    def add_document(self, doc_id: str, chunks: List[Document], metadata: Dict[str, Any]) -> None:
        """Add document and chunks to graph with optimized batch processing"""
        try:
            with self.driver.session() as session:
                # Create indexes in separate statements
                session.run("""
                    CREATE INDEX doc_id IF NOT EXISTS FOR (d:Document) ON (d.doc_id)
                """)

                session.run("""
                    CREATE INDEX doc_source IF NOT EXISTS FOR (d:Document) ON (d.source, d.source_type)
                """)

                # Create or update document node
                session.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    ON CREATE SET 
                        d.source = $source,
                        d.source_type = $source_type,
                        d.checksum = $checksum,
                        d.created_at = datetime(),
                        d += $additional_metadata
                    ON MATCH SET 
                        d.last_updated = datetime(),
                        d += $additional_metadata
                """, {
                    "doc_id": doc_id,
                    "source": metadata['source'],
                    "source_type": metadata['source_type'],
                    "checksum": metadata['checksum'],
                    "additional_metadata": {k: v for k, v in metadata.items()
                                            if k not in ['source', 'source_type', 'checksum']}
                })

                # Process chunks and entities
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}:chunk_{i}"
                    entities = self._extract_entities(chunk.page_content)

                    # Create chunk
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MERGE (c:Chunk {id: $chunk_id})
                        ON CREATE SET 
                            c.content = $content,
                            c.position = $position,
                            c.token_count = $token_count,
                            c.created_at = datetime()
                        MERGE (d)-[:HAS_CHUNK {position: $position}]->(c)
                    """, {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "content": chunk.page_content,
                        "position": i,
                        "token_count": len(chunk.page_content.split())
                    })

                    # Create entities if any exist
                    if entities:
                        session.run("""
                            UNWIND $entities as entity
                            MERGE (e:Entity {
                                name: entity.text,
                                type: entity.label
                            })
                            ON CREATE SET 
                                e.normalized_name = toLower(entity.text),
                                e.created_at = datetime()
                            SET e.last_seen = datetime()
                            WITH e, entity
                            MATCH (c:Chunk {id: $chunk_id})
                            MERGE (c)-[m:MENTIONS]->(e)
                            SET m.count = COALESCE(m.count, 0) + 1,
                                m.context = entity.context
                        """, {
                            "chunk_id": chunk_id,
                            "entities": [{
                                "text": e["text"],
                                "label": e["label"],
                                "context": chunk.page_content[
                                           max(0, e.get("start", 0) - 40):
                                           min(len(chunk.page_content), e.get("end", 0) + 40)
                                           ] if "start" in e and "end" in e else ""
                            } for e in entities]
                        })

        except Exception as e:
            self.logger.error(f"Error adding document to graph store: {str(e)}")
            raise

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document and its associated chunks from the graph database.
        Returns True if successful, False otherwise.
        """
        try:
            with self.driver.session() as session:
                result = session.run("""
                    // Match the document and related nodes
                    MATCH (d:Document {doc_id: $doc_id})
                    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (c)-[m:MENTIONS]->(e:Entity)

                    // Collect stats before deletion
                    WITH d, c, m, e,
                         count(DISTINCT d) as doc_count,
                         count(DISTINCT c) as chunk_count,
                         count(DISTINCT m) as mention_count

                    // Delete relationships and nodes
                    DELETE m, c, d

                    // Handle orphaned entities
                    WITH e, doc_count, chunk_count, mention_count
                    WHERE e IS NOT NULL
                    AND NOT EXISTS((e)<-[:MENTIONS]-())

                    // Delete orphaned entities and return stats
                    WITH e, doc_count, chunk_count, mention_count,
                         count(e) as orphaned_count
                    DELETE e

                    // Return all counts
                    RETURN doc_count as docs,
                           chunk_count as chunks,
                           mention_count as mentions,
                           orphaned_count as orphaned_entities
                """, doc_id=doc_id)

                stats = result.single()
                if stats and stats["docs"] > 0:
                    self.logger.info(
                        f"Successfully removed document {doc_id}. "
                        f"Deleted: {stats['docs']} documents, "
                        f"{stats['chunks']} chunks, "
                        f"{stats['mentions']} mentions, "
                        f"{stats['orphaned_entities']} orphaned entities"
                    )
                    return True
                else:
                    self.logger.warning(f"Document {doc_id} not found in graph database")
                    return False

        except Exception as e:
            self.logger.error(
                f"Error removing document {doc_id} from graph store: {str(e)}, "
                f"stack: {traceback.format_exc()}"
            )
            return False