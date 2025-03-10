import os
import traceback
import re
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
        """Enhanced graph search with fuzzy matching and scoring"""
        try:
            entities = self._extract_entities(query)
            if not entities:
                self.logger.warning(f"No entities extracted from query: {query}")
                return []

            self.logger.debug(f"Using entities for search: {entities}")

            with self.driver.session() as session:
                result = session.run("""
                    // Match entities with fuzzy matching
                    MATCH (e:Entity)
                    WHERE any(entity IN $entities WHERE 
                        e.normalized_name CONTAINS entity.normalized_text OR
                        entity.normalized_text CONTAINS e.normalized_name OR
                        e.name = entity.text)

                    // Find connected chunks and documents using doc_id as primary key
                    MATCH (e)<-[m:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
                    WHERE NOT EXISTS((d)<-[:REPLACES]-())

                    // Calculate relevance score
                    WITH c, d, e, m,
                         sum(CASE 
                             WHEN e.name IN $exact_names THEN 2 * m.count
                             ELSE m.count 
                         END) as relevance_score,
                         collect(DISTINCT {
                             name: e.name,
                             type: e.type,
                             context: m.context
                         }) as entity_matches

                    // Aggregate and sort results
                    WITH c, d, 
                         relevance_score,
                         entity_matches,
                         size(entity_matches) as match_count
                    ORDER BY relevance_score DESC, match_count DESC
                    LIMIT $k

                    // Return results with doc_id as primary identifier
                    RETURN 
                        c.content as content,
                        d.doc_id as doc_id,
                        d.source as source,
                        d.source_type as source_type,
                        relevance_score as graph_score,
                        entity_matches
                """, {
                    "entities": [{
                        "text": e["text"],
                        "normalized_text": e["normalized_text"]
                    } for e in entities],
                    "exact_names": [e["text"] for e in entities],
                    "k": k
                })

                documents = [Document(
                    page_content=record["content"],
                    metadata={
                        "doc_id": record["doc_id"],  # Primary identifier
                        "source": record["source"],
                        "source_type": record["source_type"],
                        "graph_score": record["graph_score"],
                        "entity_matches": record["entity_matches"],
                        "retrieval_type": "graph"
                    }
                ) for record in result]

                self.logger.debug(f"Found {len(documents)} documents through graph search")
                return documents

        except Exception as e:
            self.logger.error(f"Graph search error: {str(e)}, stack: {traceback.format_exc()}")
            return []

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced entity extraction optimized for RAG and graph storage"""
        try:
            entities = []
            doc = self.nlp(text)
            
            # 1. Named Entity Recognition with confidence scoring
            for ent in doc.ents:
                # Filter out low-quality entities
                if len(ent.text.strip()) < 2 or ent.text.strip().isnumeric():
                    continue
                    
                entities.append({
                    "text": ent.text,
                    "normalized_text": self._normalize_entity(ent.text),
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0,  # Primary entities get highest confidence
                    "type": "NER"
                })

            # 2. Key Phrase Extraction
            for np in doc.noun_chunks:
                # Filter for meaningful phrases
                if (len(np.text.split()) > 1 and  # Multi-word phrases
                    not any(word.is_stop for word in np) and  # No stop words
                    not np.text.strip().isnumeric()):  # Not just numbers
                    
                    entities.append({
                        "text": np.text,
                        "normalized_text": self._normalize_entity(np.text),
                        "label": "KEY_PHRASE",
                        "start": np.start_char,
                        "end": np.end_char,
                        "confidence": 0.8,  # Secondary confidence
                        "type": "PHRASE"
                    })

            # 3. Domain-Specific Term Extraction
            for token in doc:
                if (token.pos_ in ['PROPN', 'NOUN'] and  # Focus on important POS
                    not token.is_stop and 
                    len(token.text) > 3):
                    
                    entities.append({
                        "text": token.text,
                        "normalized_text": self._normalize_entity(token.text),
                        "label": "DOMAIN_TERM",
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                        "confidence": 0.6,  # Tertiary confidence
                        "type": "TERM"
                    })

            # 4. Entity Deduplication and Merging
            merged_entities = self._merge_entities(entities)
            
            self.logger.debug(f"Extracted {len(merged_entities)} unique entities")
            return merged_entities

        except Exception as e:
            self.logger.error(f"Entity extraction error: {str(e)}, stack: {traceback.format_exc()}")
            return []

    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text for better matching"""
        # Remove special characters and extra whitespace
        normalized = re.sub(r'[^\w\s]', ' ', text)
        normalized = ' '.join(normalized.split())
        return normalized.lower().strip()

    def _merge_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate entities"""
        merged = {}
        
        for entity in sorted(entities, key=lambda x: x['confidence'], reverse=True):
            norm_text = entity['normalized_text']
            
            if norm_text not in merged:
                merged[norm_text] = entity
            else:
                # Update existing entity if new one has higher confidence
                if entity['confidence'] > merged[norm_text]['confidence']:
                    merged[norm_text].update(entity)
        
        return list(merged.values())

    def add_document(self, doc_id: str, chunks: List[Document], metadata: Dict[str, Any]) -> None:
        """Add document and chunks to graph with optimized batch processing"""
        try:
            with self.driver.session() as session:
                # Create document node with doc_id as primary key
                session.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    ON CREATE SET 
                        d.source = $source,
                        d.source_type = $source_type,
                        d.checksum = $checksum,
                        d.created_at = datetime(),
                        d += $additional_metadata
                    ON MATCH SET 
                        d.source = $source,
                        d.source_type = $source_type,
                        d.checksum = $checksum,
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
                # First verify if document exists
                verify_result = session.run("""
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN d.doc_id as doc_id, d.source as source, d.source_type as source_type
                """, {"doc_id": doc_id})
                
                doc = verify_result.single()
                if not doc:
                    self.logger.warning(f"Document with doc_id {doc_id} not found in graph database")
                    return False
                    
                self.logger.info(f"Found document to delete: {doc}")

                # Proceed with deletion
                result = session.run("""
                    // Match the document by doc_id only
                    MATCH (d:Document {doc_id: $doc_id})
                    
                    // Get related nodes
                    OPTIONAL MATCH (d)-[r1:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (c)-[r2:MENTIONS]->(e:Entity)
                    
                    // Collect stats before deletion
                    WITH d, c, r1, r2, e,
                         count(DISTINCT d) as doc_count,
                         count(DISTINCT c) as chunk_count,
                         count(DISTINCT r2) as mention_count
                    
                    // Delete relationships first
                    DELETE r1, r2
                    
                    // Then delete nodes
                    WITH d, c, e, doc_count, chunk_count, mention_count
                    DELETE c
                    
                    // Delete document
                    WITH d, e, doc_count, chunk_count, mention_count
                    DELETE d
                    
                    // Handle orphaned entities
                    WITH e, doc_count, chunk_count, mention_count
                    WHERE e IS NOT NULL
                    AND NOT EXISTS((e)<-[:MENTIONS]-())
                    
                    // Delete orphaned entities and return stats
                    WITH e, doc_count, chunk_count, mention_count,
                         count(e) as orphaned_count
                    DELETE e
                    
                    RETURN doc_count as docs,
                           chunk_count as chunks,
                           mention_count as mentions,
                           orphaned_count as orphaned_entities
                """, {
                    "doc_id": doc_id
                })

                stats = result.single()
                if stats:
                    self.logger.info(
                        f"Deletion stats for document {doc_id}:\n"
                        f"- Documents: {stats['docs']}\n"
                        f"- Chunks: {stats['chunks']}\n"
                        f"- Mentions: {stats['mentions']}\n"
                        f"- Orphaned entities: {stats['orphaned_entities']}"
                    )
                    return stats["docs"] > 0
                else:
                    self.logger.error(f"No stats returned after deletion attempt for document {doc_id}")
                    return False

        except Exception as e:
            self.logger.error(
                f"Error removing document {doc_id} from graph store:\n"
                f"Error: {str(e)}\n"
                f"Stack: {traceback.format_exc()}"
            )
            return False