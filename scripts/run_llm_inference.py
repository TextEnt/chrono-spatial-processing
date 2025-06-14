import click
import pandas as pd
from pathlib import Path
import aisuite as ai
from dotenv import load_dotenv
from textentlib.utils import read_configuration
from textentlib.llm_utils import fetch_prompts, query_llm, llm_responses_to_dataframe

@click.command()
@click.option('--config_path', help='Path to the YAML configuration file.')
@click.option('--base_path', help='Path to the base directory for data.')
def run(config_path: str, base_path: str) -> None:
    """
    TODO: Add docstring for the function.
    """
    # Convert paths to Path objects
    config_path = Path(config_path)
    base_path = Path(base_path)

    # Read configuration
    config = read_configuration(config_path)
    settings = config['inference']
    print(f'Using configuration file: {config_path}')

    llms = settings['models']
    pregen_prompts_path = base_path / settings['pregenerated_prompts_path']
    inference_output_path = base_path / settings['responses_path']
    corpus_metadata_path = base_path / settings['corpus_metadata_path']

    # Print the list of LLMs
    click.echo("Models used for inference:")
    click.echo('\n'.join(llms))

    df_corpus = pd.read_csv(corpus_metadata_path, sep='\t')
    print(f'There are {df_corpus.shape[0]} documents to process.')

    # Filter the pre-generated prompts based on the configuration
    llm_requests = [
        prompt
        for prompt in fetch_prompts(pregen_prompts_path, df_corpus.document_id.tolist())
        if prompt.prompt_id in settings['prompts']
    ]
    print(f'Loaded {len(llm_requests)} pre-generated prompts.')

    # Initialize the AI client
    client = ai.Client()
    client.configure({"ollama": {"timeout": 600}})

    # Query LLMs and collect responses
    llm_responses = []
    reasoning_llms = ['openai:o1-mini', 'deepseek:deepseek-reasoner', 'anthropic:claude-3-7-sonnet-20250219']
    default_temperature = config['validation']['temperature']

    for model in llms:
        print('Running predictions for model:', model)
        #if 'ollama' in model:
        #    print('Ollama model detected, skipping this model.')
        #    continue
        if model in reasoning_llms:
            llm_responses += query_llm(client, model, llm_requests, inference_output_path)
        else:
            llm_responses += query_llm(client, model, llm_requests, inference_output_path, temperature=default_temperature)

    df_inference = llm_responses_to_dataframe(inference_output_path)
    df_inference.to_csv(inference_output_path / 'llm_responses.csv', sep='\t', index=False)
    print(f'LLM responses saved to: {inference_output_path / "llm_responses.csv"}')

if __name__ == '__main__':
    load_dotenv()
    run()