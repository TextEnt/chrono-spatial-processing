# Classes and functions to deal with LLM querying and post-processing of responses
import time
import re
import json
import pandas as pd
import contextlib
import aisuite
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass

JSON_PATTERN = re.compile(r"```json\n(.*?)```", re.DOTALL)
DIRECT_JSON_PATTERN = re.compile(r"\{[^}]*\}", re.DOTALL)

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
    duration_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# TODO: serialize a JSON file instead of a text file
def serialize_llm_responses(responses: List[LLMresponse], output_path: Path) -> None:

    for response in responses:

        output_dir = Path(output_path / response.document_id)

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{response.document_id}_{response.prompt_id}_{response.model_name.replace(':', '-')}.json"
        filepath = output_path / response.document_id / filename
        response_trimmed = response.response.replace('```json', '').replace('```', '').strip()

        response_dict = {
            "document_id": response.document_id,
            "prompt_id": response.prompt_id,
            "model_name": response.model_name,
            "response": response_trimmed,
            "duration_seconds": response.duration_seconds,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens
        }

        with filepath.open("w", encoding="utf-8") as file:
            file.write(json.dumps(response_dict, indent=2))
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
        
        # Capture the start time
        start_time = time.time()
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": request.prompt}])
        # Capture the end time
        end_time = time.time()
        # Calculate the duration
        duration = end_time - start_time
        print(f"Time taken to get response: {duration:.2f} seconds. Total tokens: {response.usage.total_tokens if hasattr(response, 'usage') else None}")
        
        llm_response = LLMresponse(
            document_id=request.document_id,
            prompt_id=request.prompt_id,
            prompt=request.prompt,
            model_name=model,
            response=response.choices[0].message.content,
            duration_seconds=duration,
            prompt_tokens=response.usage.prompt_tokens if hasattr(response, 'usage') else None,
            completion_tokens=response.usage.completion_tokens if hasattr(response, 'usage') else None,
            total_tokens=response.usage.total_tokens if hasattr(response, 'usage') else None,
        )
        responses.append(llm_response)
        serialize_llm_responses([llm_response], output_path)
    return responses

def try_extract_json_from_text(text: str) -> tuple[str, dict | None]:
    # function taken from https://danielvanstrien.xyz/posts/2025/deepseek/distil-deepseek-modernbert.html
    if match := JSON_PATTERN.search(text):
        json_results = match.group(1)
        with contextlib.suppress(json.JSONDecodeError):
            return text, json.loads(json_results)
    if match := DIRECT_JSON_PATTERN.search(text):
        json_text = match.group(0)
        with contextlib.suppress(json.JSONDecodeError):
            return text, json.loads(json_text)
    return text, None

def process_json_response(response_raw: str) -> Dict:
    
    output_dict = {}
    output_dict['is_response_empty'] = True if response_raw.strip() == '' else False

    try:
        response_json = json.loads(response_raw)
        output_dict['is_response_valid_json'] = True
        output_dict.update(response_json)
    except json.JSONDecodeError:
        output_dict['is_response_valid_json'] = False
        _, response_json = try_extract_json_from_text(response_raw)
        if response_json:
            output_dict.update(response_json)
    return output_dict

def process_llm_responses(llm_responses_path: Path) -> pd.DataFrame:
    # each sub-folder contains the responses for a given document
    # we need to group the responses by model so that separate dataframes can be generated
    responses = []

    # TODO: update this when the response files are serialized as JSON
    all_response_files = list(llm_responses_path.glob('*/*.txt'))
    for file_path in all_response_files:
        doc_id, prompt_id, model_id = file_path.name.replace('.txt', '').split('_')
        with file_path.open("r", encoding="utf-8") as file:
            response_raw = file.read()
        response = {
            "document_id": doc_id,
            "prompt_id": prompt_id,
            "model_id": model_id,
            "response_raw": response_raw
        }

        response_content = process_json_response(response_raw)
        response.update(response_content)
        responses.append(response)
    return pd.DataFrame(responses)

def llm_responses_to_dataframe(responses_base_path: Path) -> pd.DataFrame:
    df = process_llm_responses(responses_base_path)
    df.drop(columns=['response_raw', 'timeframe_reasoning'], inplace=True)
    
    # fusion timeframe_start and timeframe_end into a single column
    df['timeframe'] = df['timeframe_start'].astype(str) + ', ' + df['timeframe_end'].astype(str)
    df.drop(columns=['timeframe_start', 'timeframe_end'], inplace=True)

    # create a unique response ID
    df['response_id'] = df['document_id'].astype(str) + '$' + df['prompt_id'].astype(str) + '$' + df['model_id'].astype(str)
    df.set_index('response_id', inplace=True, drop=True)
    
    # rename only selected columns
    prediction_columns = ['period', 'period_reasoning', 'location', 'location_reasoning', 'location_qid', 'timeframe']
    cols = df.columns[df.columns.str.contains('|'.join(prediction_columns))]
    df.rename(columns={col: 'pred_' + col for col in df.columns if col in cols}, inplace=True)
    return df

def gt_annotations_to_dataframe(gt_base_path: Path, filename: str = 'textent-annotations - groundtruth-annotations.tsv') -> pd.DataFrame:
    df = pd.read_csv(gt_base_path / filename, sep='\t').set_index('document_id')
    df.drop(columns=['Unnamed: 11', 'Unnamed: 12', 'author', 'title', 'Anthology'], inplace=True)
    df['timeframe'] = df['timeframe_start'].astype(str) + ', ' + df['timeframe_end'].astype(str)
    df.drop(columns=['timeframe_start', 'timeframe_end'], inplace=True)
    return df.add_prefix('gt_')

def gt_metadata_to_dataframe(gt_base_path: Path, filename: str = 'textent-annotations - sample-metadata.tsv') -> pd.DataFrame:
    return pd.read_csv(gt_base_path / filename, sep='\t').set_index('document_id')

def fetch_prompts(input_path: Path, keep_document_ids: List[str]) -> List[LLMrequest]:
    """
    Fetches pre-generated prompts from the specified directory and returns a list of LLMrequest objects.

    Args:
        input_path (Path): The directory path where the prompts are located.
        keep_document_ids (List[str]): A list of document IDs to filter which files to keep.

    Returns:
        List[LLMrequest]: A list of LLMrequest objects containing the prompt ID, document ID, file path, and prompt text.
    """
    requests = []
    for file in input_path.glob(f"*/*.txt"):
        doc_id, prompt_id = file.name.split('_')
        if doc_id in keep_document_ids:
            prompt = file.read_text()
            requests.append(LLMrequest(prompt_id, doc_id, file, prompt))
    return requests