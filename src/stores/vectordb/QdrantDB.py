from typing import Any, Dict, List, Optional
from qdrant_client import AsyncQdrantClient
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
        self.api_key = settings.VECTORDB_API_KEY
        self.client = AsyncQdrantClient(
                    url=self.url,
                    api_key=self.api_key
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

    async def disconnect(self):
        if self.client:
            await self.client.close()
            self.client = None

    async def test(self) -> bool:
        if not self.client:
            await self.connect()
        try:
            await self.client.get_collections()
            return True
        except Exception as e:
            print(f"[QdrantDB]: Connection test failed: {e}")
            return False

    async def collection_exists(self, collection_name: str) -> bool:
        if not self.client:
            await self.connect()
        return await self.client.collection_exists(collection_name=collection_name)

    async def create_collection(self, collection_name: str, vector_size: int, override: bool = False) -> bool:
        if not self.client:
            await self.connect()

        exists = await self.collection_exists(collection_name)
        if exists:
            if override:
                await self.client.delete_collection(collection_name=collection_name)
            else:
                print(f"[QdrantDB]: Collection '{collection_name}' already exists.")
                return True

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=self.distance_method
            )
        )
        print(f"[QdrantDB]: Successfully created collection '{collection_name}'.")
        return True

    async def delete_collection(self, collection_name: str) -> bool:
        if not self.client:
            await self.connect()
        return await self.client.delete_collection(collection_name=collection_name)

    async def insert_records(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        if not self.client:
            await self.connect()

        qdrant_points = [
            PointStruct(
                id=item["id"],
                vector=item["vector"],
                payload=item.get("payload", {})
            )
            for item in points
        ]

        await self.client.upsert(
            collection_name=collection_name,
            points=qdrant_points
        )
        return True

    async def search(self, collection_name: str, query_vector: List[float], top_k: int = 3, query_filter: Optional[Filter] = None) :
        if not self.client:
            await self.connect()

        results = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
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

    async def search_by_text(self, collection_name: str, query_text: str, top_k: int = 3, query_filter: Optional[Filter] = None):
        query_vector = await self.embed_query(query_text)
        return await self.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            query_filter=query_filter
        )

    async def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)