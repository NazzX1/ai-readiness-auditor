from abc import ABC, abstractmethod


class BaseConnector(ABC):

    @abstractmethod
    def get_all_tables_or_collections(self):
        pass

    @abstractmethod
    def get_full_metadata(self, table_name : str):
        pass

    @abstractmethod
    def get_sample(self, table_name : str, limit : int = 10):
        pass