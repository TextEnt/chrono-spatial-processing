# Classes and functions to deal with LLM prompting

import json
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
from spacy.tokens import Doc

def build_summary_prompt(spacy_doc: Doc, prompts_base_path: Path, summaries_base_path: Path) -> str:
    """
    Builds a summary prompt based on a spaCy document.

    Args:
        spacy_doc (Doc): A spaCy document object containing the text and metadata.

    Returns:
        str: A formatted summary prompt.

    The summary is loaded from a JSON file located in the "../data/summaries" directory.
    The filename of the summary is derived from the 'document_id' stored in the user_data attribute of the spaCy document.
    """

    # load base prompt
    summary_prompt_path = prompts_base_path / "summary_prompt.txt"
    with open(summary_prompt_path, "r") as file:
        base_prompt = file.read()

    # load the pre-computed summary from its JSON file
    doc_summary_path = summaries_base_path / f"{spacy_doc.user_data['document_id']}_summary.json"

    try:    
        with doc_summary_path.open('r', encoding='utf-8') as file:
            summary = json.load(file)
    except:
        return None

    # JSON to pretty string
    summary_as_string = json.dumps(summary, indent=2, ensure_ascii=False)
    return base_prompt.format(document_summary=summary_as_string)

def build_excerpt_prompt(spacy_doc: Doc, prompts_base_path: Path, summaries_base_path: Path, excerpt_length: int = 400) -> str:
    # load the pre-computed summary from its JSON file
    doc_summary_path = summaries_base_path / f"{spacy_doc.user_data['document_id']}_summary.json"

    # load excerpt prompt
    excerpt_prompt_path = prompts_base_path / "excerpt_prompt.txt"
    with open(excerpt_prompt_path, "r") as file:
        prompt = file.read()

    # load the pre-computed summary from its JSON file
    doc_summary_path = summaries_base_path / f"{spacy_doc.user_data['document_id']}_summary.json"
    
    try:    
        with doc_summary_path.open('r', encoding='utf-8') as file:
            summary = json.load(file)
    except:
        #print(f'No summary for document {spacy_doc.user_data["document_id"]}')
        return None

    text_length = len(spacy_doc.text)
    mid_point = text_length // 2
    left_boundary = mid_point - (excerpt_length // 2) 
    right_boundary = mid_point + (excerpt_length // 2)
    excerpt = spacy_doc.text[left_boundary:right_boundary]

    # JSON to pretty string
    json_doc = {
        'metadata': summary['metadata'],
        'excerpt': excerpt
    }
    json_doc_as_string = json.dumps(json_doc, indent=2, ensure_ascii=False)
    return prompt.format(document=json_doc_as_string, excerpt_length=excerpt_length)

def build_metadata_prompt(spacy_doc: Doc, prompts_base_path: Path, summaries_base_path: Path) -> str:
    # load the pre-computed summary from its JSON file
    doc_summary_path = summaries_base_path / f"{spacy_doc.user_data['document_id']}_summary.json"

    # load metadata prompt
    metadata_prompt_path = prompts_base_path / "metadata_prompt.txt"
    with open(metadata_prompt_path, "r") as file:
        metadata_prompt = file.read()

    # load the pre-computed summary from its JSON file    
    try:    
        with doc_summary_path.open('r', encoding='utf-8') as file:
            summary = json.load(file)
    except:
        #print(f'No summary for document {spacy_doc.user_data["document_id"]}')
        return None

    # JSON to pretty string
    metadata = {'metadata': summary['metadata']}
    metadata_as_string = json.dumps(metadata, indent=2, ensure_ascii=False)
    return metadata_prompt.format(document_metadata=metadata_as_string)

def build_prompts(spacy_doc: Doc, prompts_base_path: Path, summaries_base_path: Path) -> List[Tuple[str, str]]:
    """
    Builds prompts based on a spaCy document.

    Args:
        spacy_doc (Doc): A spaCy document object containing the text and metadata.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains a prompt ID and its text.    
    """
    return [
        ('prompt-summary', build_summary_prompt(spacy_doc, prompts_base_path, summaries_base_path)),
        ('prompt-metadata', build_metadata_prompt(spacy_doc, prompts_base_path, summaries_base_path)),
        ('prompt-excerpt', build_excerpt_prompt(spacy_doc, prompts_base_path, summaries_base_path)),
    ]

def pre_generate_prompts(spacy_docs: List[Doc], prompts_base_path: Path, summaries_base_path: Path, output_path: Path) -> None:
    """
    Pre-generates prompts for a list of spaCy documents and writes them to the specified output path.
    Args:
        spacy_docs (List[Doc]): A list of spaCy Doc objects, each containing user data with a "document_id".
        output_path (Path): The path to the directory where the prompts will be saved.
    Returns:
        None
    This function iterates over each spaCy document, generates prompts using the `build_prompts` function,
    and writes each prompt to a text file in a directory named after the document's ID. If a prompt cannot
    be generated, an error message is appended to the `problems` list, which is printed at the end.
    """

    problems = []

    for spacy_doc in tqdm(spacy_docs, desc=f"Pre-generating prompts (destination: {output_path})"):
        doc_id = spacy_doc.user_data["document_id"]
        prompts = build_prompts(spacy_doc, prompts_base_path, summaries_base_path)

        # Define the path to the directory
        directory_path = output_path / doc_id

        # Check if the directory exists
        if not directory_path.exists():
            directory_path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist

        for prompt_id, prompt in prompts:
            if prompt:
                #print(f"Writing prompt {prompt_id} for document {doc_id}")
                with open(output_path / doc_id / f"{doc_id}_{prompt_id}.txt", "w") as file:
                    file.write(prompt)
            else:
                problems.append(f'There was a problem with generating prompt {prompt_id} for document {doc_id}')
    
    print("\n".join(problems))