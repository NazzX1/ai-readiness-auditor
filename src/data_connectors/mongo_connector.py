from pymongo import MongoClient
from src.data_connectors.base_connector import BaseConnector
from src.models.schema_models import ColumnInfo, TableMetadata




class MongoConnector(BaseConnector):
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]


    def get_all_tables_or_collections(self):
        return self.db.list_collection_names()
    

    def get_full_metadata(self):
        metadata = {}

        collections = self.get_all_tables_or_collections()

        for collection in collections:
            metadata[collection] = self.get_collection_metadata(collection)
        

        return metadata
    


    def get_collection_metadata(self, collection_name : str):
        stats = self._get_collection_stats(collection_name)

        fields = self._get_fields(collection_name)

        return TableMetadata(
            table_name=collection_name,
            columns=fields,
            foreign_keys=[],
            row_count=stats["row_count"],
            size_bytes=stats["size_bytes"]
        )





    def _get_collection_stats(self, collection_name : str):
        stats = self.db.command("collStats", collection_name)
        return{
            "row_count" : stats.get('count', 0),
            "size_bytes" : stats.get("size", 0)
        }
    

    def _get_fields(self, collection_name : str, sample_size = 20):
        collection = self.db[collection_name]
        samples = list(collection.aggregate([{"$sample": {"size": sample_size}}]))
        
        all_keys = set()
        for doc in samples:
            all_keys.update(doc.keys())
            
        return [ColumnInfo(name=key, data_type="dynamic", is_nullable=True, comment="") 
                for key in sorted(all_keys)]
    
    def get_sample_from_collection(self, collection_name : str, percentage : float = 0.1):

        collection = self.db[collection_name]


        count = collection.estimated_document_count()

        sample_count = max(1, int(count * percentage))

        return list(collection.aggregate([{"$sample": {"size": sample_count}}]))
    

    def get_sample(self):
        collections = self.get_all_tables_or_collections()
        samples = {}
        for collection in collections:
            samples[collection] = self.get_sample_from_collection(collection)
        
        return samples





