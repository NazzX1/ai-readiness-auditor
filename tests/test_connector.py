import pytest
import os
from src.data_connectors.postgres_connector import PostgresConnector
from src.data_connectors.mongo_connector import MongoConnector
from pprint import pprint



@pytest.fixture(scope="session")
def postgres_connector():
    uri = f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@localhost:5432/{os.getenv('POSTGRES_DB_NAME', 'postgres')}"
    connector = PostgresConnector(uri)
    print(connector)
    return connector

@pytest.fixture(scope="session")
def mongo_connector():
    uri = f"mongodb://{os.getenv('MONGO_USER', 'admin')}:{os.getenv('MONGO_PASSWORD', 'password')}@localhost:27017"
    connector = MongoConnector(uri, db_name=os.getenv('MONGO_DB_NAME', 'test_db'))
    print(connector)
    yield connector


def test_postgres_connection(postgres_connector):
    tables = postgres_connector.get_all_tables_or_collections()
    assert isinstance(tables, list)

def test_mongo_connection(mongo_connector):
    collections = mongo_connector.get_all_tables_or_collections()
    assert isinstance(collections, list)

def test_postgres_metadata_structure(postgres_connector):
    tables = postgres_connector.get_all_tables_or_collections()
    if tables:
        metadata = postgres_connector.get_table_metadata(tables[0])
        print(metadata)
        assert metadata.table_name == tables[0]
        assert isinstance(metadata.columns, list)
    metadatas = postgres_connector.get_full_metadata()
    pprint(metadatas)

def test_mongo_metadata_structure(mongo_connector):
    collections = mongo_connector.get_all_tables_or_collections()
    if collections:
        metadata = mongo_connector.get_collection_metadata(collections[0])
        print("=" * 6)
        print(metadata)
        assert metadata.table_name is not None
        assert metadata.row_count >= 0
    
    metadatas = mongo_connector.get_full_metadata()
    pprint(metadatas)