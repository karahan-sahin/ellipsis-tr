from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Enum

# Enums

class EllipsisType(Enum):
    
    OBJECT_DROP = "OBJECT_DROP"
    SUBJECT_DROP = "SUBJECT_DROP"
    VP_ELLIPSIS = "VP_ELLIPSIS"
    NP_ELLIPSIS = "NP_ELLIPSIS"

class HumanAnnotationStatus(Enum):
    
    IN_ANNOTATION = "STATUS_CODE_100"
    IN_REVIEW = "STATUS_CODE_300"
    REVIEWED = "STATUS_CODE_200"
    REJECTED = "STATUS_CODE_400"
    ACCEPTED = "STATUS_CODE_500"


# Models

class Annotation(BaseModel):
    
    annotation_type: str
    annotator_id: str
    annotator_type: str
    elliptical_type: EllipsisType
    span: List[str]
    correlate: List[str]
    sentence_quality: bool
    annotator_note: str
    annotation_time: str
    
    
class Instance(BaseModel):
    
    candidate_id: str
    candidate_text: str
    provenance: str
    acquisition_method: str
    parse: dict
    annotation: List[Annotation]
    next_context: List[str]
    previous_context: List[str]
    human_annotation_status: HumanAnnotationStatus
        


class ExperimentType(Enum):
    
    TYPE_CLASSIFICATION = 'TYPE_CLASSIFICATION'
    DISCRIMINITIVE_SPAN_CLASSIFICATION  = 'DISCRIMINITIVE_SPAN_CLASSIFICATION'
    EXTRACTIVE_SPAN_CLASSIFICATION  = 'EXTRACTIVE_SPAN_CLASSIFICATION'
    CORRELATE_GENERATION = 'CORRELATE_GENERATION'
    
class Environment(BaseModel):
    
    EXPERIMENT_SUFFIX: str = 'v0.1.2'
    
    TASK_TYPE: ExperimentType = ExperimentType.TYPE_CLASSIFICATION
    
    OUTPUT_MODEL_DIR: str = 'drive/MyDrive/Ellipsis Experimental Models'
    OUTPUT_LOG_DIR: str = 'drive/MyDrive/Ellipsis Project/Experiments/Experimental Logs/'
    
    
    PRETRAINED_MODEL_NAME: str = 'dbmbz/bert-turkish-cased-128k'
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 20
    LEARNING_RATE: int = 1e-5
    WEIGHT_DECAY: int = 1e-5
    
    AUGMENT: bool = True
    
