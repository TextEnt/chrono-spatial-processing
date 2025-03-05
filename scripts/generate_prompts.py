"""Description: Command-line script to generate document prompts from a Spacy corpus."""

# input: path to spacy corpus, path to output directory for doc summaries, path to output directory for prompts
# sample size (optional)
# clear output directories before writing (optional)
# output: document summaries and prompts

import click
import random
from pathlib import Path
from textentlib.summary import generate_document_summaries
from textentlib.prompting import pre_generate_prompts

PROMPTS_BASE_PATH = Path("/Users/mromanel/Documents/UniGe-TextEnt/chrono-spatial-processing/data/prompts/")

@click.command()
@click.option('--spacy-corpus-path', help='Path to the serialized Spacy corpus.')
@click.option('--output-summaries-path', help='Path to a folder where to store document summaries.')
@click.option('--output-prompts-path', help='Path to a folder where to store document summaries.')
@click.option('--sample-size', default=None, help='Number of files to sample (default=None).')
def main(spacy_corpus_path, output_summaries_path, output_prompts_path, sample_size):
    from textentlib.utils import load_or_create_corpus, nlp_model_fr
    #print(nlp_model_fr.pipe_names)
    
    output_prompts_path = Path(output_prompts_path)
    output_summaries_path = Path(output_summaries_path)
    
    # Load spacy corpus
    spacy_corpus = load_or_create_corpus(spacy_corpus_path)
    docs = spacy_corpus.get_docs(nlp_model_fr.vocab)
    docs = list(docs)
    
    if sample_size:
        sampled_docs = random.sample(docs, sample_size)
    else:
        sampled_docs = docs
    print(f'Working with {len(sampled_docs)} documents')
    
    # generate document summaries if `output_summaries_path` does not exist
    if not output_summaries_path.exists():
        generate_document_summaries(sampled_docs, output_summaries_path)
    else:
        print(f"Document summaries already exist at {output_summaries_path}; skipping summary generation")

    # we can not know if prompts were generated, so we generate them anew
    pre_generate_prompts(sampled_docs, PROMPTS_BASE_PATH, output_summaries_path, output_prompts_path)

if __name__ == '__main__':
    main()