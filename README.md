# Chrono-spatial processing for the TextEnt project

This repository contains code and data originating from the [TextEnt](https://www.unige.ch/lettres/humanites-numeriques/recherche/projets/projets-de-la-chaire/textent) project to perform the prediction of fictional time and space of French theatre plays (17C).

## Installation

```bash
pip install -r requirements.txt
pip install .
python -m spacy download fr_core_news_lg
```

## Pipeline

This pipeline is implemented as a series of Python scripts that can be found in `./scripts/`; for documentation on how to use each command, see instructions and configuration in [`Makefile`](./Makefile).

### Pre-processing

The pre-processing step (see `make tei2spacy`) takes as input [a corpus of TEI/XML files](https://github.com/TextEnt/TheatreLFSV2) with NER information and transforms them into a SpaCy corpus with some basic metadata. Successively, Entity Linking (EL) is performed by means of the entity-fishing tool via its SpaCy wrapper (TODO add Makefile step).  

### Generation of summaries and prompts

For each document, we generate automatically a short document summary which contains information about the named entities that are most frequenly mentioned in the text (see `make generate-prompts`). These summaries are then used in the prompts passed to the LLMs; overall, we experimented with three types of prompts: one containing only the document's metadata; one containing metadata and a random excerpt of 400 words, sampled from the middle of the text; and one with metadata and the automatically generated document summary.

### Benchmark data

Data for validation and evaluation can be found in `groundtruth`; document annotations are in the file `textent-annotations - sample-metadata.tsv`, while manual annotations are stored in `textent-annotations - groundtruth-annotations.tsv`. There is a small set of validation documents that can be used for few-shot training: their metadata contain `keep_fine_tuning == True`; all the other documents (where `keep_fine_tuning == False`) are used for evaluation.

### LLM-judge validation

In order to chose a model for the LLM-judge evaluation, we first compare the scores produced by several models against human scores in order to ensure that they are aligned. To this end, we run the LLM predictions on the validation documents (see section above) with `make ...`, and then ask all candidate judge models to assign scores to these predictions (the code for this is still in a notebook). At the same time, we asked two human annotators to assign scores to the same predictions by using a small tool built for this purpose (see [`textent-evaluator-UI`](https://github.com/TextEnt/textent-evaluator-UI)). We then compute the inter annotator agreement (IAA) between the LLM-judges' scores and the human scores in order to select one model (the code for this is still in a notebook).

### Evaluation

Once we have identified a suitable LLM-judge, we benchmark several LLMs in a zero-shot setting (meaning that no model training nor fine-tuning is performed). For the list of benchmarked models, see `data/confi.yaml` under the section `evaluation.models`. To perform the evaluation against the test set, just run `make run-llm-judge-evaluation` and specify the name of an LLM to be used as a judge. This script will produce also an evaluation report, where models' predictions, groundtruth annotations and assigned scores can be easily compared (For an example, see `data/evaluation/evaluation_report.md`). 

### Inference

Based on the evaluation results, we can then select an LLM and use it for inference on a larger corpus (see `make ...`). This step assumes that pre-processing and pre-generation of prompts were already performed. For an example configuration see `data/confi.yaml` under the section `inference`. As an example, the folder `data/llm_inference/` contains all the predictions produced by the model [`ollama:mistral-small3.1:latest`](https://ollama.com/library/mistral-small3.1) on the documents listed in `data/textent_corpus.tsv`. All model's prediction are then compiled into a TSV file [`./data/llm_inference/llm_responses.tsv`](./data/llm_inference/llm_responses.tsv).

## Acknowledgements
Code and data in this repository were produced in the context of the project _The Geographic Horizon of writers_ (PIs Simon Gabay and Nicola Carboni), funded by the Swiss National Science Foundation under the Spark grant [220833](https://data.snf.ch/grants/grant/220833).
