SHELL:=/bin/bash

BASE_PATH?='/Users/mromanello/Documents/chrono-spatial-processing'
CONFIG_FILE?='./data/config.yaml'
CORPUS_PATH?='../TheatreLFSV2/'
SPACY_CORPUS_PATH?='data/corpus_24022025.spacy'
LLM_JUDGE_MODEL?='openai:o1-mini'

tei2spacy:
	python scripts/tei2spacy.py --corpus-path $(CORPUS_PATH) --spacy-corpus-path $(SPACY_CORPUS_PATH)


# pre-generate prompts for all documents in the spacy corpus
generate-prompts:
	python scripts/generate_prompts.py --spacy-corpus-path=$(SPACY_CORPUS_PATH) 
		--output-summaries-path=data/summaries/
		--output-prompts-path=data/prompts/pregenerated2/

run-llm-prediction:
	python scripts/run_llm_predictions.py --config_path=data/config.yaml 
		--base_path=/Users/mromanel/Documents/UniGe-TextEnt/chrono-spatial-processing/ --split=eval

run-llm-judge-evaluation:
	python scripts/run_llm_judge_evaluation.py --config-path=data/config.yaml 
		--base-path $(BASE_PATH) --llm-judge=$(LLM_JUDGE_MODEL)