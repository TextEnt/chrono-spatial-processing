# Classes and functions to deal with LLM querying and post-processing of responses
import aisuite
from typing import List
from pathlib import Path
from dataclasses import dataclass

import aisuite as ai

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

def serialize_llm_responses(responses: List[LLMresponse], output_path: Path) -> None:

    for response in responses:

        output_dir = Path(output_path / response.document_id)

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{response.document_id}_{response.prompt_id}_{response.model_name.replace(':', '-')}.txt"
        filepath = output_path / response.document_id / filename
        response_trimmed = response.response.replace('```json', '').replace('```', '').strip()

        with filepath.open("w", encoding="utf-8") as file:
            file.write(response_trimmed)
    return

def query_llm(client: aisuite.Client, model: str, requests: List[LLMrequest], output_path: Path) -> List[LLMresponse]:
    # pass over the requests to a given model and gather the responses
    responses = []
    for request in requests:
        # Avoid asking the model, if an answer file already exists
        filename = f"{request.document_id}_{request.prompt_id}_{model.replace(':', '-')}.txt"
        filepath = output_path / request.document_id / filename
        if filepath.exists():
            #print(f"Found file {filepath} for document {request.document_id}")
            print(f"Skipping request for document {request.document_id}[{request.prompt_id}] using model {model} as it already exists")
            continue
            
        print(f"Processing prompt {request.prompt_id} for document {request.document_id} using model {model}")
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": request.prompt}])
        llm_response = LLMresponse(
            document_id=request.document_id,
            prompt_id=request.prompt_id,
            prompt=request.prompt,
            model_name=model,
            response=response.choices[0].message.content
        )
        responses.append(llm_response)
        serialize_llm_responses([llm_response], output_path)
    return responses