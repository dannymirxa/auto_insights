from typing import List, Optional, Dict
from typing_extensions import Annotated, TypeAlias, Union
from annotated_types import MinLen
from pydantic import BaseModel, Field
from pydantic import field_validator


class SummarySuccess(BaseModel):
    summary: str = Field(alias='summary', description='summary of insights')

class SummaryInvalidRequest(BaseModel):
    error_message: str

SummaryResponse: TypeAlias = Union[SummarySuccess, SummaryInvalidRequest]

class QnASuccess(BaseModel):
    survey_detail: str = Field(alias='survey_detail', description='details of the survey')
    answer: str = Field(alias='answer', description='answer regarding insights')

class QnAInvalidRequest(BaseModel):
    error_message: str

QnAResponse: TypeAlias = Union[QnASuccess, QnAInvalidRequest]