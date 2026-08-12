from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
)

from langchain_ollama import OllamaEmbeddings
from src.stores.vectordb.VectorDBInterface import VectorDBInterface
from src.stores.vectordb.VectorDBEnums import DistanceMethodsEnums
from src.helpers.config import Settings


class QdrantDB(VectorDBInterface):
    def __init__(self, settings: Settings):
        self.url = settings.VECTORDB_URL
        self.client = QdrantClient(
                    url=self.url,
                )
        self.distance_method = settings.VECTORDB_DISTANCE_METHOD
        self.embedding_model = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.EMBEDDING_BASE_URL
        )

        if self.distance_method == DistanceMethodsEnums.COSINE.value:
            self.distance_method = Distance.COSINE
        elif self.distance_method == DistanceMethodsEnums.DOT.value:
            self.distance_method = Distance.DOT
        else:
            self.distance_method = Distance.COSINE

    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None

    def test(self) -> bool:
        if not self.client:
            self.connect()
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            print(f"[QdrantDB]: Connection test failed: {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        if not self.client:
            self.connect()
        return self.client.collection_exists(collection_name=collection_name)

    def create_collection(self, collection_name: str, vector_size: int, override: bool = False) -> bool:
        if not self.client:
            self.connect()

        exists = self.collection_exists(collection_name)
        if exists:
            if override:
                self.client.delete_collection(collection_name=collection_name)
            else:
                print(f"[QdrantDB]: Collection '{collection_name}' already exists.")
                return True

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=self.distance_method
            )
        )
        print(f"[QdrantDB]: Successfully created collection '{collection_name}'.")
        return True

    def delete_collection(self, collection_name: str) -> bool:
        if not self.client:
            self.connect()
        return self.client.delete_collection(collection_name=collection_name)

    def insert_records(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        if not self.client:
            self.connect()

        qdrant_points = [
            PointStruct(
                id=item["id"],
                vector=item["vector"],
                payload=item.get("payload", {})
            )
            for item in points
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=qdrant_points
        )
        return True

    def search(self, collection_name: str, query_vector: List[float], top_k: int = 3, query_filter: Optional[Filter] = None) :
        if not self.client:
            self.connect()

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        

        search_hits = []
        for res in results:
            search_hits.append({
                "id": res.id,
                "score": res.score,
                "payload": res.payload
            })

        return search_hits

    def search_by_text(self, collection_name: str, query_text: str, top_k: int = 3, query_filter: Optional[Filter] = None):
        query_vector = self.embed_query(query_text)
        return self.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            query_filter=query_filter
        )

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)