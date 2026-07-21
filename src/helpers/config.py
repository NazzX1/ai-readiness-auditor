import http

from pydantic_settings import BaseSettings





class Settings(BaseSettings):
    OLLAMA_MODEL : str
    OLLAMA_BASE_URL : str
    PROFILER_TEMP : float
    ML_PLANNER_TEMP : float
    
        
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


    class Config:
        env_file = ".env"



def get_settings():
    return Settings()





