from src.helpers.config import Settings
from src.stores.vectordb.VectorDBEnums import VectorDBenums




class VectorDBFactory:
    def __init__(self, settings : Settings):
        self.settings = settings
    

    def create(self, provider : str):
        if provider == VectorDBenums.QDRANT.value:
            return
        

        return None