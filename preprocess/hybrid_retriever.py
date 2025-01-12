from typing import List, Dict
from langchain_core.documents import Document

class HybridRetriever:
    def __init__(self, vector_store, graph_store):
        self.vector_store = vector_store
        self.graph_store = graph_store

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        # Get vector similarity results
        vector_results = self.vector_store.similarity_search(query, k=k)
        
        # Get graph-based results
        graph_results = self.graph_store.find_related_chunks(query, k=k)
        
        # Combine and deduplicate results
        all_results = vector_results + [
            Document(page_content=content) 
            for content in graph_results
        ]
        
        # Deduplicate and return top k
        seen = set()
        unique_results = []
        for doc in all_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_results.append(doc)
        
        return unique_results[:k] 