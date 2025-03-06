# Classes and functions to deal with LLM querying and post-processing of responses

from pathlib import Path
from dataclasses import dataclass

@dataclass
class LLMrequest:
    prompt_id: str
    document_id: str
    prompt_path: Path
    prompt: str

@dataclass
class LLMresponse:
    document_id: str
    prompt_id: str
    prompt: str
    model_name: str
    response: str