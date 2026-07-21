from abc import abstractmethod, ABC


class BaseFileReader(ABC):
    def __init__(self, file_path : str):
        self.file_path = file_path
    
    @abstractmethod
    def read(self):
        raise NotImplementedError("Subclasses must implement it")
    
    def parse(self):
        return None
    
    def filter(self, kept_keys):
        return None
    

    def get_full_metadata(self):
        pass