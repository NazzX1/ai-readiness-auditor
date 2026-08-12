from typing import Annotated, TypedDict, Optional, Union, Any
from dataclasses import dataclass
from src.models.schema_models import ColumnInfo, TaskCategory, Scope, SensitivityLevel
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
    sample : Optional[list]

    data_lineage : list[str]
    sensitivity_lvl : SensitivityLevel
    tags  : list[str]
    governance_assessment : Any


    size_bytes : int
    created_at : str


    @classmethod
    def from_dict(cls, data : dict) -> "DataMetadata":
        return cls(
            name = data["name"],
            description = data["description"],
            file_type = data["file_type"],
            modality = data["modality"],
            structure = data["structure"],
            sample = data.get("sample"),
            data_lineage = data.get("data_lineage", []),
            sensitivity_lvl = SensitivityLevel(data["sensitivity_lvl"]),
            tags = data.get("tags", []),
            governance_assessment = data.get("governance_assessment"),
            size_bytes = data["size_bytes"],
            created_at = data["created_at"]
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "file_type": self.file_type,
            "modality": self.modality,
            "structure": self.structure,
            "sample": self.sample,
            "data_lineage": self.data_lineage,
            "sensitivity_lvl": self.sensitivity_lvl.value if isinstance(self.sensitivity_lvl, SensitivityLevel) else self.sensitivity_lvl,
            "tags": self.tags,
            "governance_assessment": self.governance_assessment,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at
        }



@dataclass
class ContextualPayload:
    task : Optional[str] = None
    domain : Optional[str] = None
    target_column : Optional[str] = None
    additional_info : Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "domain": self.domain,
            "target_column": self.target_column,
            "additional_info": self.additional_info
        }




class AgentState(TypedDict):
    messages = Annotated[list[BaseMessage], add_messages]
    session_id : int
    data_source : str
    data_source_params : str
    data_metadata : DataMetadata
    scope : Scope
    domain : str
    task_type : TaskCategory
    contextual_payload : ContextualPayload
    error : str

    evaluation_plan : Any
    reasoning_trace : Any
    evaluation_plan : Any
    llm_analysis : Any

    execution_results : Any







def initial_state(session_id : str, data_source : str, **data_source_params):
    return {
        "messages" : [],
        "session_id" : session_id,
        "data_source" : data_source,
        "data_source_params" : data_source_params,
        "data_metadata" : None,
        "domain" :  None,
        "task_type" : None,
        "contextual_payload" : "",
        "error" : "",        
        "evaluation_plan" : None,
        "reasoning_trace" : None,
        "evaluation_plan" : None,
        "llm_analysis" : None,
        "execution_results" : None
    }
    






