import aisuite as ai
import click
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from textentlib.llm_utils import LLMrequest, query_llm_judge, process_llm_judge_responses
from textentlib.llm_utils import prepare_evaluation_dataframe, add_prompt
from textentlib.utils import read_configuration

@click.command()
@click.option('--config-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help="Path to the YAML configuration file.")
@click.option('--base-path', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help="Path to the base directory for data.")
@click.option('--llm-judge', type=str, required=True, help="The LLM judge to use (e.g., 'gpt-3.5-turbo', 'gpt-4').")
def run_llm_judge_evaluation(config_path, base_path, llm_judge):
    """
    Command-line script to run the LLM-judge evaluation.
    """
    # Load environment variables
    load_dotenv()

    # Convert paths to Path objects
    config_path = Path(config_path)
    base_path = Path(base_path)
    
    # Read configuration
    config = read_configuration(config_path)['evaluation']
    gt_path = base_path / config['groundtruth_path']

    evaluation_docs, df_evaluation_data = prepare_evaluation_dataframe(
        llm_response_path=Path(config['responses_path']),
        gt_annotations_path=gt_path,
        gt_metadata_path=gt_path,
        split="eval"
    )
    click.echo(f"Found {df_evaluation_data.shape[0]} predictions on {len(evaluation_docs)} documents.")

    df_evaluation_data['llm_judge_prompt'] = df_evaluation_data.apply(
        add_prompt, args=([Path(config['llm_judge_prompt_path'])]),
        axis=1
    )

    # build the requests to pass on to the LLM-judge
    llm_judge_requests =[
        LLMrequest(
            prompt_id='llm_judge_prompt',
            document_id=response_id,
            prompt_path=None,
            prompt=item['llm_judge_prompt']
        )
        for response_id, item in df_evaluation_data[['llm_judge_prompt']].to_dict(orient='index').items()
    ]

    llm_judge_responses = query_llm_judge(ai.Client(), llm_judge, llm_judge_requests)
    scores  = process_llm_judge_responses(llm_judge_responses)
    scores_dir = base_path / config['scores_output_path']
    scores_dir.mkdir(parents=True, exist_ok=True) # Ensure the directory for scores exists
    scores_tsv_path = base_path / config['scores_output_path'] / f"evaluation_{llm_judge}_scores.tsv"
    pd.DataFrame(scores).to_csv(f"{scores_tsv_path}", sep='\t')
    print(f"Scores saved to {scores_tsv_path}")

if __name__ == '__main__':
    run_llm_judge_evaluation()