import click
import shutil
from pathlib import Path
import aisuite as ai
from textentlib.utils import read_configuration
from textentlib.llm_utils import fetch_prompts, prepare_evaluation_dataframe, query_llm

@click.command()
@click.argument('config_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument('base_path', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
def run(config_path: str, base_path: str, split_to_process: str):
    """
    Run validation of LLM-judge evaluation.

    CONFIG_PATH: Path to the YAML configuration file.
    BASE_PATH: Base path for data.
    """
    # Convert paths to Path objects
    config_path = Path(config_path)
    base_path = Path(base_path)

    # Read configuration
    config = read_configuration(config_path)
    llms = config['validation']['models']
    gt_path = base_path / config['validation']['groundtruth_path']
    pregen_prompts_path = base_path / config['validation']['pregenerated_prompts_path']
    validation_llm_responses_path = base_path / config['validation']['responses_path']

    # Print the list of LLMs
    click.echo("Models to process:")
    click.echo('\n'.join(llms))

    # Prepare validation documents and prompts
    validation_docs, df_validation_data = prepare_evaluation_dataframe(
        llm_response_path=validation_llm_responses_path,
        gt_annotations_path=gt_path,
        gt_metadata_path=gt_path,
        split=split_to_process
    )

    llm_requests = fetch_prompts(pregen_prompts_path, validation_docs)

    # Clean up the LLM responses directory
    def clean_up_directory(directory_path: Path) -> None:
        for item in directory_path.iterdir():
            if item.is_dir():
                click.echo(f'Removed folder {item} and all its contents')
                shutil.rmtree(item)
            else:
                click.echo(f'Removed file {item}')
                item.unlink()

    # Uncomment the following line to clean up the directory before running
    # clean_up_directory(validation_llm_responses_path)

    # Initialize the AI client
    client = ai.Client()
    client.configure({"ollama": {"timeout": 600}})

    # Query LLMs and collect responses
    llm_responses = []
    reasoning_llms = ['openai:o1-mini', 'deepseek:deepseek-reasoner', 'anthropic:claude-3-7-sonnet-20250219']
    default_temperature = config['validation']['temperature']

    for model in llms:
        if model.startswith('ollama'):
            click.echo(f'Skipping {model}')
            continue
        if model in reasoning_llms:
            llm_responses += query_llm(client, model, llm_requests, validation_llm_responses_path)
        else:
            llm_responses += query_llm(client, model, llm_requests, validation_llm_responses_path, temperature=default_temperature)

    # Read LLM responses (again, to get the latest ones) into a dataframe to export
    validation_docs, df_validation_data = prepare_evaluation_dataframe(
        llm_response_path=validation_llm_responses_path,
        gt_annotations_path=gt_path,
        gt_metadata_path=gt_path,
        split=split_to_process
    )
    df_validation_data.to_csv(base_path / f'data/{split_to_process}_predictions.tsv', sep='\t')
    click.echo("Predictions completed.")

if __name__ == '__main__':
    run()