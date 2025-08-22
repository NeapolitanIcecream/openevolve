# OpenEvolve

Commit-based evolutionary coding agent: a practical, end-to-end system for evolving entire codebases inside a Git repository using LLMs, MAP-Elites and an island model, and process-based parallelism.

## About This Fork

This fork modernizes OpenEvolve into a full code agent that performs commit-based evolution directly in a Git repo:

- Commit is the unit of evolution: each organism is a Git commit that may modify multiple files
- Fast similarity via MinHash on cleaned diffs from a configurable root commit
- KV-cache-friendly long sessions: stable system prefix, incremental per-iteration context, on-demand compression
- Flexible on-the-fly tool invocation: read, edit, glob/grep, ls, and submit; evaluator is wrapped as a tool
- True parallelism using per-worker Git worktrees and a process pool
- MAP-Elites with island model for quality-diversity; periodic migration across islands

## How It Works

1. Sampling: From the current island, sample a parent plus diverse inspirations from the ProgramDatabase (MAP-Elites + islands)
2. Prompt/session: A stable system message (KV-cache) is kept; per-iteration context includes goal, parent/inspiration diffs, and parent metrics
3. Tool loop: The LLM uses built-in tools to read and edit files, then calls the submit tool once (with commit_message) to finish the iteration
4. Commit: Changes in the dedicated worktree are committed; metrics may be summarized in the commit message
5. Update DB: ProgramDatabase derives cleaned diffs, MinHash signatures, feature coordinates, and updates MAP-Elites and island bests
6. Migration/checkpointing: Periodic migration maintains diversity; checkpoints save DB state and the best commit

## Key Capabilities

- Commit-based evolution with Git worktrees and safe isolation
- KV-cache-aware long sessions with on-demand compression
- LLM tool loop with: edit, read_file, read_many_files, glob, grep, ls, submit
- MAP-Elites + island model (quality-diversity); default feature dimensions: complexity and diversity
- MinHash-based diversity/similarity; configurable signature parameters
- Artifacts side-channel for rich execution feedback
- Checkpoints and resuming; live visualization web UI

## Installation

Via uv (recommended):

```bash
git clone https://github.com/NeapolitanIcecream/openevolve.git
cd openevolve

# Install uv (macOS):
#   brew install uv
# Or via shell:
#   curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual env and install project dependencies
uv sync --all-projects
```

Alternative:

```bash
pip install -e .
```

Set your LLM provider (any OpenAI-compatible API):

```bash
export OPENAI_API_KEY=your-api-key
# optionally:
export OPENAI_API_BASE=https://your-provider-endpoint/v1
```

## Quick Start

Prepare a Git repository containing the code you want to evolve, and an evaluator script exposing `evaluate(repo_root)` (or `evaluate()` with cwd set by the system). The evaluator must return a dictionary of metrics; for best results include a `combined_score`. The LLM will use the `submit` tool to trigger evaluation and provide its `commit_message`.

### CLI

```bash
uv run openevolve-run /absolute/path/to/repo /absolute/path/to/evaluator.py \
  --config /absolute/path/to/config.yaml \
  --iterations 200
```

Useful flags (override config at runtime):

- `--root-commit`: baseline commit/branch for diffs (default: HEAD)
- `--evolution-target`: natural-language goal to steer evolution
- `--similarity-threshold`: signature similarity threshold (0–1)
- `--api-base`, `--primary-model`, `--secondary-model`, `--write-tool-model`
- Session controls: `--session-max-tokens`, `--session-compress-threshold`, `--recent-history-tokens`, `--compression-model`
- Checkpoint/resume: `--checkpoint`, output dir: `--output`, logging: `--log-level`

Resume from a checkpoint directory created under `<repo>/openevolve_output/checkpoints`:

```bash
uv run openevolve-run /repo /evaluator.py --checkpoint /repo/openevolve_output/checkpoints/checkpoint_500 --iterations 100
```

### Python API

```python
import asyncio
from openevolve import OpenEvolve

async def main():
    evo = OpenEvolve(
        git_repo_path="/abs/path/to/repo",
        evaluation_file="/abs/path/to/evaluator.py",
        config_path="/abs/path/to/config.yaml",
    )
    best = await evo.run(iterations=200)
    if best:
        print("Best metrics:", best.metrics)

asyncio.run(main())
```

## Configuration

The configuration closely matches the dataclasses in `openevolve/config.py`. Minimal example:

```yaml
max_iterations: 200
random_seed: 42

llm:
  defaults:
    api_base: "${OPENAI_API_BASE:-https://api.openai.com/v1}"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.7
    top_p: 0.95
    max_tokens: 4096
  models:
    - name: "gpt-4o-mini"   # primary evolution model
      weight: 1.0
  write_tool_model_name: "gpt-4o-mini"
  tool_loop_max_steps: 30

prompt:
  system_message: "You are an expert software agent working inside a git worktree..."
  include_artifacts: true
  max_artifact_bytes: 20480
  max_inspirations: 2
  session_max_tokens: 120000
  session_compress_threshold: 80000
  recent_history_tokens: 30000
  compression_model_name: null

database:
  git_repo_path: "/abs/path/to/repo"
  root_commit: "HEAD"               # or a specific SHA/branch
  evolution_target: "Improve throughput without regressing correctness"
  population_size: 1000
  num_islands: 5
  migration_interval: 50
  migration_rate: 0.1
  feature_dimensions: ["complexity", "diversity"]
  feature_bins: 10
  minhash_num_perm: 64
  minhash_shingle_len: 5
  commit_message_template: "OpenEvolve iteration {iteration} {commit_message} {metrics}"
  commit_message_max_metrics: 6

evaluator:
  timeout: 300
  max_retries: 3
  parallel_evaluations: 1
  enable_artifacts: true
  require_evaluate_before_commit: true
```

Notes:

- The database combines MAP-Elites with an island model. Built-in default feature dimensions are `complexity` and `diversity`; you may also list evaluator metrics as feature dimensions. If a configured dimension is not available, the system raises an error with available keys.
- If `combined_score` exists in evaluator output, it becomes the primary fitness; otherwise the system uses the average of numeric metrics.

## Built-in Tools (OpenAI function tools)

- `read_file` / `read_many_files`: Read file contents under the worktree root
- `glob`, `grep`, `ls`: Discover files and search code
- `edit`: Make safe, minimal edits to files (used to implement multi-file commits)
- `submit`: Run your evaluator on the current worktree and return `{metrics, commit_message}` as JSON; ends the iteration

The tool registry lives under `openevolve/tools`. The LLM chooses tools freely; you should avoid dynamically adding/removing tool definitions mid-session to preserve KV cache hits.

## Checkpoints and Outputs

Under `<repo>/openevolve_output/` the system saves:

- `checkpoints/checkpoint_<N>/` containing the serialized database, prompt logs (optional), and `best_commit.txt`
- `best/` with `best_commit.txt` and `best_program_info.json`

## Visualization

Interactive web UI to browse the evolution tree:

```bash
uv run scripts/visualizer.py
# or with a specific checkpoint
uv run scripts/visualizer.py --path /repo/openevolve_output/checkpoints/checkpoint_1000
```

## Maintenance / Cleanup

Use the cleanup utility to remove branches and objects created during evolution when a run is finished or you want to reclaim local space:

```bash
# Remove all evolution branches (default prefix oe/it_) and run Git GC (dry-run by default)
uv run openevolve-clean --repo /abs/path/to/repo --all

# Actually execute (non dry-run)
uv run openevolve-clean --repo /abs/path/to/repo --all --yes

# Keep a specific commit (if no evolution branch has this commit as HEAD, a keep branch will be created) and delete other evolution branches
uv run openevolve-clean --repo /repo --keep <commit_sha> --yes

# Options
# - Custom evolution branch prefix
uv run openevolve-clean --repo /repo --all --prefix oe/it_ --yes
# - Remove .openevolve/worktrees (enabled by default; use --no-remove-worktrees to disable)
uv run openevolve-clean --repo /repo --all --no-remove-worktrees --yes
# - Remove the output directory
uv run openevolve-clean --repo /repo --all --remove-output --yes
# - Tag the kept commit
uv run openevolve-clean --repo /repo --keep <sha> --tag openevolve/selected --yes
# - Also delete matching branches on a remote (must explicitly provide remote)
uv run openevolve-clean --repo /repo --all --remote origin --yes
```

Notes:
- Branches are matched by the evolution branch prefix (default `oe/it_`).
- In `--keep <commit>` mode, any evolution branch whose HEAD equals the commit is preserved; if none, a new `oe/keep/<short_sha>` branch is created pointing at the commit. You can add `--tag` to create/update a tag for extra safety.
- Dry-run by default. Add `--yes` to actually apply changes; use `--dry-run` to force a preview.
- Cleanup removes worktrees under `.openevolve/worktrees` first, then deletes branches, and finally runs `git gc` to reclaim objects.

## FAQ

- Do I need a clean Git repo? Yes. The system manages per-worker worktrees and will hard-reset them each iteration. Your main repository is not modified except for commits the agent creates.
- Can I use non-OpenAI providers? Yes, any OpenAI-compatible endpoint via `OPENAI_API_BASE` or `llm.defaults.api_base`.
- How is similarity computed? MinHash signatures of cleaned diffs from `root_commit` to the program commit; diversity is `1 - Jaccard(similarity)`.
