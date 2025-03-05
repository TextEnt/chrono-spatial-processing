"""Description: Command-line script to convert TEI files (with NER information) to Spacy corpus format"""

import click
from tqdm import tqdm
from pathlib import Path
from spacy.tokens import DocBin
from textentlib.utils import tei2spacy_simple, nlp_model_fr, sample_files, print_corpus_summary

@click.command()
@click.option('--spacy-corpus-path', default='./data/corpus.spacy', help='Path to the serialized Spacy corpus.')
@click.option('--corpus-path', default='../TheatreLFSV2-downloaded/', help='Path to the corpus directory.')
@click.option('--sample-size', default=None, help='Number of files to sample (default=None).')
def main(spacy_corpus_path, corpus_path, sample_size):
    if Path(spacy_corpus_path).exists():
        spacy_corpus = DocBin(store_user_data=True).from_disk(spacy_corpus_path)
        print(f"Loaded serialized spacy corpus from {spacy_corpus_path}")
    else:
        spacy_corpus = DocBin(store_user_data=True)

    print_corpus_summary(spacy_corpus, nlp_model_fr)

    already_processed_files = set([Path(doc.user_data['path']) for doc in spacy_corpus.get_docs(nlp_model_fr.vocab)])

    corpus_basedir = Path(corpus_path) / 'NER'
    if sample_size:
        print(f"Sampling {sample_size} files from {corpus_basedir}")
        sampled_files = sample_files(corpus_basedir, sample_size, already_processed_files)
    else:
        print(f"Processing all files from {corpus_basedir}")
        sampled_files = set(corpus_basedir.iterdir()) - already_processed_files

    # there should not be files in the sample that have already been processed
    assert len(set(sampled_files) - already_processed_files) == len(sampled_files)

    docs = [tei2spacy_simple(file) for file in tqdm(sampled_files)]

    for doc in docs:
        spacy_corpus.add(doc)
    print_corpus_summary(spacy_corpus, nlp_model_fr)

    spacy_corpus.to_disk(spacy_corpus_path)

if __name__ == '__main__':
    main()