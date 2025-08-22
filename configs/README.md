# OpenEvolve Configuration (Commit-Based Evolution)

This directory contains configuration templates and examples for the commit-based evolution workflow.

## Files

### `default_config.yaml`
Complete template with all supported options and sane defaults:
- LLM ensemble with shared `defaults` and per-model weights
- Long-session/KV-cache friendly prompt settings
- Commit-based evolution (git) parameters and island model
- Evaluator settings including commit gating

Use this file as a starting point for your own runs.

### `island_config_example.yaml`
Ready-to-run example showcasing a balanced island configuration plus commit-based settings.

### `island_examples.yaml`
Profiles for different exploration strategies you can merge into your config:
- Maximum Diversity (broad exploration)
- Focused Exploration (deep local search)
- Balanced (recommended default)
- Quick Exploration (small-scale prototyping)
- Large-Scale (extensive search)

## Key Commit-Based Parameters

```yaml
database:
  # Baseline used to compute diffs for signatures and snapshots
  root_commit: "HEAD"

  # Natural language goal for the run; influences prompts and selection
  evolution_target: "Improve correctness and performance while maintaining API."

  # Controls how aggressively we treat two commits as near-duplicates
  signature_similarity_threshold: 0.8
```

## Session Management (KV-cache Friendly)

```yaml
prompt:
  session_max_tokens: 120000
  session_compress_threshold: 80000
  recent_history_tokens: 30000
```

## LLM Ensemble

```yaml
llm:
  defaults:
    api_base: "https://api.openai.com/v1"
    api_key: null
  models:
    - name: "gpt-4o-mini"
      weight: 0.8
    - name: "gpt-4o"
      weight: 0.2
```

## Island Model

```yaml
database:
  num_islands: 5
  migration_interval: 50
  migration_rate: 0.1
```

Guidelines:
- num_islands: 3-10 for most problems (more = more diversity)
- migration_interval: 25-100 (higher = more independent evolution)
- migration_rate: 0.05-0.2 (higher = faster knowledge sharing)

## CLI Usage

Run OpenEvolve using your repository and evaluator:

```bash
uv run -m openevolve.cli /path/to/repo /path/to/evaluator.py \
  --config configs/default_config.yaml \
  --root-commit <baseline_sha_or_branch> \
  --evolution-target "Improve correctness and performance while maintaining API." \
  --write-tool-model gpt-4o-mini \
  --compression-model gpt-4o-mini
```

Alternatively, use only the config file (set `database.root_commit` etc. inside YAML):

```bash
uv run -m openevolve.cli /path/to/repo /path/to/evaluator.py --config my_config.yaml
```
