from abc import ABC, abstractmethod



class VectorDBInterface(ABC):

    def connect(self):
        pass


    def disconnect(self):
        pass



    def list_all_collections(self):
        pass


    def insert_many(self):
        pass


    def create_collection(self, collection_name : str):
        pass


    def search_by_vector(self):
        pass