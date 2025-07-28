# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Overview

OpenEvolve is an open-source implementation of Google DeepMind's AlphaEvolve system - an evolutionary coding agent that uses LLMs to optimize code through iterative evolution. The framework can evolve code in multiple languages (Python, R, Rust, etc.) for tasks like scientific computing, optimization, and algorithm discovery.

This version extends the original concept to perform whole-codebase evolution, where each evolutionary step is a Git commit that can modify multiple files. It operates as a tool-using agent, leveraging LLMs to iteratively improve a software project within a Git repository.

## Essential Commands

### Development Setup
```bash
# First, create the virtual environment
uv venv

# Then, sync your environment with the lockfile
uv sync --dev

# Or use Makefile
make install
```

### Running Tests
```bash
# Run all tests
uv run python -m unittest discover tests

# Or use Makefile
make test
```

### Code Formatting
```bash
# Format with Black
uv run python -m black openevolve examples tests scripts

# Or use Makefile
make lint
```

### Running OpenEvolve
```bash
# Basic evolution run
uv run python openevolve-run.py path/to/repo --config path/to/config.yaml --iterations 1000

# Resume from checkpoint
uv run python openevolve-run.py path/to/repo \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50
```

### Visualization
```bash
# View evolution tree
uv run python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

## High-Level Architecture

### Core Components

1. **Controller (`openevolve/controller.py`)**: Main orchestrator that manages the evolution process using ProcessPoolExecutor for parallel iteration execution.

2. **Database (`openevolve/database.py`)**: Implements MAP-Elites algorithm with island-based evolution:
   - Commits mapped to multi-dimensional feature grid
   - Multiple isolated populations (islands) evolve independently
   - Periodic migration between islands prevents convergence
   - Tracks absolute best commit separately

3. **Evaluator (`openevolve/evaluator.py`)**: Cascade evaluation pattern:
   - Stage 1: Quick validation
   - Stage 2: Basic performance testing  
   - Stage 3: Comprehensive evaluation
   - Commits must pass thresholds at each stage

4. **LLM Integration (`openevolve/llm/`)**: Ensemble approach with multiple models, configurable weights, and async generation with retry logic.

5. **Tools (`openevolve/tools`)**: Built-in tools that models use to interact with local environment, access information, and perform actions.

### Key Architectural Patterns

- **Island-Based Evolution**: Multiple populations evolve separately with periodic migration
- **MAP-Elites**: Maintains diversity by mapping commits to feature grid cells
- **Artifact System**: Side-channel for commits to return debugging data, stored as JSON or files
- **Process Worker Pattern**: Each iteration runs in fresh process with database snapshot
- **Double-Selection**: Commits for inspiration differ from those shown to LLM
- **Lazy Migration**: Islands migrate based on generation counts, not iterations

### Configuration

YAML-based configuration with hierarchical structure:
- LLM models and parameters
- Database and island settings
- Evaluation parameters

### Important Patterns

1. **Checkpoint/Resume**: Automatic saving of entire system state with seamless resume capability
2. **Parallel Evaluation**: Multiple commits evaluated concurrently via TaskPool
3. **Error Resilience**: Individual failures don't crash system - extensive retry logic and timeout protection
4. **Prompt Engineering**: Template-based system with context-aware building and evolution history

### Development Notes

- Python >=3.9 required
- Uses OpenAI-compatible APIs for LLM integration
- Tests use unittest framework
- Black for code formatting
- Artifacts threshold: Small (<10KB) stored in DB, large saved to disk
- Process workers load database snapshots for true parallelism
