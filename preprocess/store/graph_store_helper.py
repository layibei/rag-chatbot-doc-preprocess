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
    def __init__(self, graphDatabase: GraphDatabase):
        self.logger = logger
        self.driver = graphDatabase
        
        # Load spaCy model from local path

        model_path = Path(os.join(BASE_DIR, "../../models/spacy/en_core_web_md"))
        if not model_path.exists():
            raise RuntimeError(
                "spaCy model not found. Please run scripts/download_spacy_model.py first"
            )
        self.nlp = spacy.load(str(model_path))
        
    def add_document(self, doc_id: str, metadata: Dict, chunks: List[Document]):
        try:
            with self.driver.session() as session:
                # Create document node
                session.run("""
                    MERGE (d:Document {id: $doc_id})
                    SET d += $metadata
                """, doc_id=doc_id, metadata=metadata)
                
                # Process chunks and extract entities
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    entities = self._extract_entities(chunk.page_content)
                    
                    # Create chunk node and relationship
                    session.run("""
                        MATCH (d:Document {id: $doc_id})
                        MERGE (c:Chunk {id: $chunk_id})
                        SET c.content = $content,
                            c.position = $position
                        MERGE (d)-[:HAS_CHUNK]->(c)
                    """, doc_id=doc_id, chunk_id=chunk_id, 
                         content=chunk.page_content, position=i)
                    
                    # Create entity nodes and relationships
                    for entity in entities:
                        session.run("""
                            MERGE (e:Entity {name: $name, type: $type})
                            WITH e
                            MATCH (c:Chunk {id: $chunk_id})
                            MERGE (c)-[:MENTIONS]->(e)
                        """, name=entity["text"], type=entity["label"], 
                             chunk_id=chunk_id)
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

    def find_related_chunks(self, query: str, k: int = 3) -> List[str]:
        """Find related chunks using graph traversal"""
        try:
            entities = self._extract_entities(query)
            
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name IN $entity_names
                    MATCH (e)<-[:MENTIONS]-(c:Chunk)
                    WITH c, count(DISTINCT e) as relevance
                    ORDER BY relevance DESC
                    LIMIT $k
                    RETURN c.content as content
                """, entity_names=[e["text"] for e in entities], k=k)
                
                return [record["content"] for record in result]
        except Exception as e:
            self.logger.error(f"Error finding related chunks: {str(e)}")
            return [] 