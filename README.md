# Chrono-spatial processing for the TextEnt project

## Commands memo

### Pre-generate prompts

```bash
python scripts/generate_prompts.py --spacy-corpus-path=data/corpus_24022025.spacy --output-summaries-path=data/summaries/ --output-prompts-path=data/prompts/pregenerated2/
```

### Run LLM prediction on eval set

```bash
python scripts/run_llm_predictions.py --config_path=data/config.yaml --base_path=/Users/mromanel/Documents/UniGe-TextEnt/chrono-spatial-processing/ --split=eval
```

### Run LLM-Judge evaluation

```bash
python scripts/run_llm_judge_evaluation.py --config-path=data/config.yaml --base-pcessing --llm-judge=openai:o1-minie-TextEnt/chrono-spatial-proc
```

## Acknowledgements
Code and data in this repository were produced in the context of the project _The Geographic Horizon of writers_ (PIs Simon Gabay and Nicola Carboni), funded by the Swiss National Science Foundation under the Spark grant [220833](https://data.snf.ch/grants/grant/220833).
