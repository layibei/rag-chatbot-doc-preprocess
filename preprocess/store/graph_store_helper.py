import os
from typing import List, Dict
from neo4j import GraphDatabase
from langchain_core.documents import Document
import spacy
from pathlib import Path
from utils.logging_util import logger

# Get the absolute path of the current file
CURRENT_FILE_PATH = os.path.abspath(__file__)
# Get the directory containing the current file
BASE_DIR = os.path.dirname(CURRENT_FILE_PATH)

class GraphStoreHelper:
    def __init__(self, graph_db: GraphDatabase):
        self.logger = logger
        self.driver = graph_db
        
        if not self.driver:
            self.logger.warning("No Neo4j driver provided - graph store operations will be disabled")
            return
        
        try:
            # Load spaCy model from local path
            model_path = Path(os.path.join(BASE_DIR, "../../models/spacy/en_core_web_md"))
            if not model_path.exists():
                raise RuntimeError(
                    "spaCy model not found. Please run scripts/download_spacy_model.py first"
                )
            self.nlp = spacy.load(str(model_path))
            self.logger.info("Successfully initialized GraphStoreHelper")
        except Exception as e:
            self.logger.error(f"Failed to initialize GraphStoreHelper: {str(e)}")
            self.driver = None

    def _generate_document_id(self, metadata: Dict) -> str:
        """Generate a meaningful document ID from metadata"""
        return f"{metadata['source_type']}:{metadata['source']}:{metadata['checksum']}"

    def add_document(self, doc_id: str, metadata: Dict, chunks: List[Document]):
        """Add document to graph store"""
        if not self.driver:
            raise RuntimeError("Graph store is not initialized or disabled")
        
        try:
            self.logger.info(f"Adding document {doc_id} to graph store")
            
            # Generate meaningful document ID
            document_id = self._generate_document_id(metadata)
            
            with self.driver.session() as session:
                # Create document node with composite ID and version tracking
                session.run("""
                    MERGE (d:Document {composite_id: $composite_id})
                    SET d.original_id = $original_id,
                        d.source = $source,
                        d.source_type = $source_type,
                        d.checksum = $checksum,
                        d.last_updated = datetime(),
                        d += $additional_metadata
                """, 
                    composite_id=document_id,
                    original_id=doc_id,  # Keep original ID for reference
                    source=metadata['source'],
                    source_type=metadata['source_type'],
                    checksum=metadata['checksum'],
                    additional_metadata={k: v for k, v in metadata.items() 
                                      if k not in ['source', 'source_type', 'checksum']}
                )
                
                # Process chunks with versioning support
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{document_id}:chunk_{i}"
                    entities = self._extract_entities(chunk.page_content)
                    
                    # Create chunk node with better metadata
                    session.run("""
                        MATCH (d:Document {composite_id: $doc_id})
                        MERGE (c:Chunk {id: $chunk_id})
                        SET c.content = $content,
                            c.position = $position,
                            c.token_count = $token_count,
                            c.created_at = datetime()
                        MERGE (d)-[:HAS_CHUNK {position: $position}]->(c)
                    """, 
                        doc_id=document_id,
                        chunk_id=chunk_id,
                        content=chunk.page_content,
                        position=i,
                        token_count=len(chunk.page_content.split())
                    )
                    
                    # Create entity nodes with context
                    for entity in entities:
                        session.run("""
                            MERGE (e:Entity {
                                name: $name,
                                type: $type,
                                normalized_name: $normalized_name
                            })
                            SET e.last_seen = datetime()
                            WITH e
                            MATCH (c:Chunk {id: $chunk_id})
                            MERGE (c)-[m:MENTIONS]->(e)
                            SET m.count = COALESCE(m.count, 0) + 1,
                                m.context = $context
                        """, 
                            name=entity["text"],
                            type=entity["label"],
                            normalized_name=entity["text"].lower(),
                            chunk_id=chunk_id,
                            context=text[max(0, entity["start"]-40):min(len(text), entity["end"]+40)]
                            if "start" in entity and "end" in entity else ""
                        )
                
                # Add document version relationship if previous version exists
                session.run("""
                    MATCH (current:Document {composite_id: $composite_id})
                    MATCH (prev:Document)
                    WHERE prev.source = $source 
                      AND prev.source_type = $source_type
                      AND prev.checksum <> $checksum
                      AND prev.composite_id <> $composite_id
                    MERGE (current)-[:REPLACES]->(prev)
                """,
                    composite_id=document_id,
                    source=metadata['source'],
                    source_type=metadata['source_type'],
                    checksum=metadata['checksum']
                )
                
        except Exception as e:
            self.logger.error(f"Error adding document to graph store: {str(e)}")
            raise

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
            self.logger.error(f"Error finding related chunks: {str(e)}")
            return [] 