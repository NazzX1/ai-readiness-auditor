from src.stores.vectordb.VectorDBInterface import VectorDBInterface
from qdrant_client.models import Distance
from src.helpers.config import Settings
from qdrant_client import AsyncQdrantClient
from src.stores.vectordb.VectorDBEnums import DistanceMethodsEnums


class QdrantDB(VectorDBInterface):
    def __init__(self, settings : Settings):
        self.client = None
        self.url = settings.VECTORDB_URL
        self.api_key = settings.VECTORDB_API_KEY
        self.distance_method = settings.VECTORDB_DISTANCE_METHOD

        if self.distance_method == DistanceMethodsEnums.COSINE.value:
            self.distance_method = Distance.COSINE
        elif self.distance_method == DistanceMethodsEnums.DOT.value:
            self.distance_method = Distance.DOT

    

    def connect(self):
        self.client = AsyncQdrantClient(
            url= self.url,
            api_key= self.api_key
        )
    

    def disconnect(self):
        self.client = None
    

    def test(self):
        pass

        
