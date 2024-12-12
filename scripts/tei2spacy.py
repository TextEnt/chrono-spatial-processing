import sys
import textentlib
import click
import dask.bag as db
from pathlib import Path
from spacy.tokens import DocBin
from textentlib.utils import tei2spacy, nlp_model_fr, sample_files, print_corpus_summary
from dask.distributed import Client, LocalCluster

@click.command()
@click.option('--dask-scheduler', default=None, help='Address of a running Dask cluster/scheduler.')
@click.option('--spacy-corpus-path', default='./data/corpus.spacy', help='Path to the serialized Spacy corpus.')
@click.option('--corpus-path', default='../TheatreLFSV2-downloaded/', help='Path to the corpus directory.')
@click.option('--sample-size', default=10, help='Number of files to sample.')
def main(spacy_corpus_path, corpus_path, sample_size, dask_scheduler):
    
    if Path(spacy_corpus_path).exists():
        spacy_corpus = DocBin(store_user_data=True).from_disk(spacy_corpus_path)
        print(f"Loaded serialized spacy corpus from {spacy_corpus_path}")
    else:
        spacy_corpus = DocBin(store_user_data=True)

    print_corpus_summary(spacy_corpus, nlp_model_fr)

    already_processed_files = set([Path(doc.user_data['path']) for doc in spacy_corpus.get_docs(nlp_model_fr.vocab)])

    corpus_basedir = Path(corpus_path)
    sampled_files = sample_files(Path(corpus_basedir / 'NER'), sample_size, already_processed_files)

    # there should not be files in the sample that have already been processed
    assert len(set(sampled_files) - already_processed_files) == len(sampled_files)

    # setting up Dask stuff for parallel processing
    dask_client = Client(dask_scheduler) if dask_scheduler else Client(LocalCluster())

    entity_projection = True
    disable_pb = True
    docs = db.from_sequence(sampled_files).map(tei2spacy, entity_projection, disable_pb).compute()

    for doc in docs:
        spacy_corpus.add(doc)

    print_corpus_summary(spacy_corpus, nlp_model_fr)

    spacy_corpus.to_disk(spacy_corpus_path)

if __name__ == '__main__':
    main()