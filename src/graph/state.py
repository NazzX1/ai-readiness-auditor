from typing import Annotated, TypedDict, Optional, Union, Any
from dataclasses import dataclass
from src.models.schema_models import ColumnInfo, TaskCategory, Domain, SensitivityLevel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



@dataclass
class StructuredData:
    tables : dict[str, list[ColumnInfo]]
    row_count : int
    size_bytes : int


@dataclass
class SemiStructuredData:
    objects_keys : dict[str, Any]
    element_count : int

    

@dataclass
class UnstructuredData:
    language : str
    tokens_count : int
    summary : Optional[str]

@dataclass
class MediaData:
    pass


@dataclass
class DataMetadata:
    name : str
    description : str

    file_type : str
    modality : str
    structure :  Union[StructuredData, SemiStructuredData, UnstructuredData, MediaData]

    data_lineage : list[str]
    sensitivity_lvl : SensitivityLevel
    tags  : list[str]

    size_bytes : int
    created_at : str






class AgentState(TypedDict):
    messages = Annotated[list[BaseMessage], add_messages]
    session_id : int
    data_source : str
    data_source_params : str
    data_metadata : DataMetadata
    domain : Domain
    task_type : TaskCategory
    contextual_payload : str






def initial_state(session_id : str, data_source : str, **data_source_params):
    return {
        "messages" : [],
        "session_id" : session_id,
        "data_source" : data_source,
        "data_source_params" : data_source_params,
        "data_metadata" : None,
        "domain" :  None,
        "task_type" : None,
        "contextual_payload" : ""
    }
    






