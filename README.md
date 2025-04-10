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