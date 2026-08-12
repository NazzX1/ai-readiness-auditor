import http

from pydantic_settings import BaseSettings





class Settings(BaseSettings):
    OLLAMA_MODEL : str
    OLLAMA_BASE_URL : str
    PROFILER_TEMP : float
    ML_PLANNER_TEMP : float
    EXECUTOR_TEMP : float
    EVALUATOR_TEMP : float

    EMBEDDING_MODEL : str
    EMBEDDING_BASE_URL : str



    
        
    UPLOADED_DATA : str
    CHECKPOINT_DB : str
    
    
    POSTGRES_USER : str
    POSTGRES_PASSWORD : str
    POSTGRES_DB_NAME : str
    
    
    MONGO_USER : str
    MONGO_PASSWORD : str
    MONGO_DB_NAME : str



    VECTORDB : str 
    VECTORDB_API_KEY : str
    VECTORDB_URL : str
    VECTORDB_DISTANCE_METHOD : str


    NEO4J_URI : str
    NEO4J_USER : str
    NEO4J_PASSWORD : str

    REDIS_HOST : str
    REDIS_PORT : str
    REDIS_PASSWORD :str
    REDIS_DB : int


    class Config:
        env_file = ".env"



def get_settings():
    return Settings()





