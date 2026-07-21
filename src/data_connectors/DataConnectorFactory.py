from data_connectors.DataConnectorEnums import DataConnectorEnums
from src.data_connectors.mongo_connector import MongoConnector
from src.data_connectors.postgres_connector import PostgresConnector
from src.data_connectors.file_connector import FileConnector

class DataConnectorFactory:

    def getConnector(dsrc: str, **kwargs):
        if dsrc == DataConnectorEnums.POSTGRES:
            return PostgresConnector(kwargs.get('uri'))
        elif dsrc == DataConnectorEnums.MONGO:
            return MongoConnector(kwargs.get('uri'), kwargs.get('db_name'))
        elif dsrc == DataConnectorEnums.DEFAULT:
            return FileConnector(kwargs.get('file_path'))
        else:
            NotImplemented(f"{dsrc} is not supported yet")


