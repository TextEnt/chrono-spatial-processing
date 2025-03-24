# Classes and functions to deal with LLM querying and post-processing of responses
import time
import re
import json
import pandas as pd
import contextlib
import aisuite
from typing import List, Dict, Union
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

def query_llm(client: aisuite.Client, model: str, requests: List[LLMrequest], output_path: Path, temperature: float = 0.2) -> List[LLMresponse]:
    # pass over the requests to a given model and gather the responses
    responses = []
    for request in requests:
        # Avoid asking the model, if an answer file already exists
        filename = f"{request.document_id}_{request.prompt_id}_{model.replace(':', '-')}.json"
        filepath = output_path / request.document_id / filename
        if filepath.exists():
            #print(f"Found file {filepath} for document {request.document_id}")
            print(f"Skipping request for document {request.document_id}[{request.prompt_id}] using model {model} as it already exists")
            continue
            
        print(f"Processing prompt {request.prompt_id} for document {request.document_id} using model {model}")
        
        # Capture the start time
        start_time = time.time()
        # TODO: for certain models do not set the temperature (e.g. o1-mini)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=temperature
        )
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

def query_llm_judge(client: aisuite.Client, model: str, requests: List[LLMrequest], temperature: float = 0.2) -> List[LLMresponse]:
    # pass over the requests to a given model and gather the responses
    responses = []
    for request in requests:
        print(f"Processing prompt {request.prompt_id} for document {request.document_id} using model {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=temperature
        )
        llm_response = LLMresponse(
            document_id=request.document_id,
            prompt_id=request.prompt_id,
            prompt=request.prompt,
            model_name=model,
            response=response.choices[0].message.content,
            duration_seconds=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )
        responses.append(llm_response)
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

def process_json_response(response_path: Path) -> Dict:
    output_dict = {}
    
    with response_path.open("r", encoding="utf-8") as file:
        response_dict = json.loads(file.read())

    output_dict['response_path'] = response_path
    output_dict.update(response_dict)
    response_raw = response_dict['response']
    
    try:
        response_json = json.loads(response_raw)
        output_dict['is_response_valid_json'] = True
        output_dict['is_valid_json_recovered'] = False
        output_dict.update(response_json)
    except json.JSONDecodeError:
        output_dict['is_response_valid_json'] = False
        _, response_json = try_extract_json_from_text(response_raw)
        if response_json:
            output_dict['is_valid_json_recovered'] = True
            output_dict.update(response_json)
        else:
            output_dict['is_valid_json_recovered'] = False
    return output_dict

def process_llm_responses(llm_responses_path: Path) -> pd.DataFrame:
    # each sub-folder contains the responses for a given document
    responses = []
    all_response_files = list(llm_responses_path.glob('*/*.json'))
    for file_path in all_response_files:
        responses.append(process_json_response(file_path))
    return pd.DataFrame(responses)

def llm_responses_to_dataframe(responses_base_path: Path) -> pd.DataFrame:
    df = process_llm_responses(responses_base_path)
    df.drop(columns=['response'], inplace=True)
    
    # fusion timeframe_start and timeframe_end into a single column
    df['timeframe'] = df['timeframe_start'].astype(str) + ', ' + df['timeframe_end'].astype(str)
    df.drop(columns=['timeframe_start', 'timeframe_end'], inplace=True)

    # create a unique response ID
    df['response_id'] = df['document_id'].astype(str) + '$' + df['prompt_id'].astype(str) + '$' + df['model_name'].astype(str)
    df.set_index('response_id', inplace=True, drop=True)
    
    # rename only selected columns
    prediction_columns = ['period', 'period_reasoning', 'location', 'location_reasoning', 'location_qid', 'timeframe']
    cols = df.columns[df.columns.str.contains('|'.join(prediction_columns))]
    df.rename(columns={col: 'pred_' + col for col in df.columns if col in cols}, inplace=True)
    return df

def gt_annotations_to_dataframe(gt_base_path: Path, filename: str = 'textent-annotations - groundtruth-annotations.tsv') -> pd.DataFrame:
    df = pd.read_csv(gt_base_path / filename, sep='\t').set_index('document_id')
    try:
        df.drop(columns=['Unnamed: 11', 'Unnamed: 12', 'author', 'title', 'Anthology'], inplace=True)
    except KeyError as e:
        print(e)
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

def prepare_evaluation_dataframe(
        llm_response_path: Path,
        gt_metadata_path: Path,
        gt_annotations_path: Path,
        split: str = "test"
    ) -> Union[List[str], pd.DataFrame]:
    """
    Prepares a dataframe for evaluating LLM responses against ground truth annotations.
        This function processes and combines data from LLM responses, ground truth metadata, 
        and ground truth annotations to create a structured dataframe for evaluation. It also 
        filters and massages the data based on the specified split.
        Args:
            llm_response_path (Path): Path to the file containing LLM responses.
            gt_metadata_path (Path): Path to the file containing ground truth metadata.
            gt_annotations_path (Path): Path to the file containing ground truth annotations.
            split (str, optional): Specifies the data split to use. 
                - "test": Includes documents marked for fine-tuning.
                - "eval": Includes documents not marked for fine-tuning.
                Defaults to "test".
        Returns:
            Union[List[str], pd.DataFrame]: 
                - A list of document IDs included in the selected split.
                - A pandas DataFrame containing the evaluation data with relevant columns.
        Raises:
            ValueError: If an invalid split value is provided.
        Notes:
            - The function filters out documents that are marked as excluded or not annotated.
            - It merges LLM responses with ground truth annotations and metadata.
            - Additional columns are added with default values for scoring purposes.
            - The resulting dataframe is reordered and some columns are dropped for clarity.
    """
    metadata_columns_to_keep = ['author', 'title', 'publication_date', 'document_length', 'keep_fine_tuning']

    df_llm_responses = llm_responses_to_dataframe(llm_response_path)
    df_gt_annotations = gt_annotations_to_dataframe(gt_annotations_path)
    df_gt_metadata = gt_metadata_to_dataframe(gt_metadata_path)

    # filter out documents that are marked as to be excluded or that were not annotated
    df_annotated_docs = df_gt_metadata[(df_gt_metadata.exclude == 0) & (df_gt_metadata.annotated == 1)][metadata_columns_to_keep]

    if split == 'test':
        df_sample_docs = df_annotated_docs[df_annotated_docs.keep_fine_tuning == 1]
    elif split == 'eval':
        df_sample_docs = df_annotated_docs[df_annotated_docs.keep_fine_tuning != 1]
    else:
        raise ValueError(f"Invalid split value: {split}")
    
    # merging 
    df_sample_gt = df_sample_docs.join(df_gt_annotations, how='inner')
    df_eval_data = df_llm_responses.merge(df_sample_gt, left_on='document_id', right_index=True)

    # define new columns with default values
    df_eval_data['score_period_string'] = None
    df_eval_data['score_period_timeframe'] = None
    df_eval_data['score_location_string'] = None
    df_eval_data['score_location_qid'] = None

    # reorder columns and drop some
    display_columns = [
        'prompt_id',
        'model_name',
        'document_id',
        'author',
        'title',
        'publication_date',
        'document_length',
        'keep_fine_tuning',
        'gt_period',
        'pred_period',
        'score_period_string',
        'gt_timeframe',
        'pred_timeframe',
        'score_period_timeframe',
        'gt_period_reason',
        'pred_period_reasoning',
        'gt_preferred_location',
        'gt_accepted_locations',
        'pred_location',
        'score_location_string',
        'gt_preferred_location_QID',
        'gt_acceptable_location_QIDs',
        'pred_location_qid',
        'score_location_qid',
        'gt_location_reason',
        'pred_location_reasoning',
    ]
    return df_sample_docs.index.to_list(), df_eval_data[display_columns]