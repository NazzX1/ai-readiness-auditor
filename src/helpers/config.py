from pydantic_settings import BaseSettings





class Settings(BaseSettings):
    UPLOADED_DATA : str
    CHECKPOINT_DB : str

    OLLAMA_MODEL : str
    OLLAMA_BASE_URL : str
    PROFILER_TEMP : float



    class Config:
        env_file = ".env"



def get_settings():
    return Settings()





