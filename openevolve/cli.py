"""
Command-line interface for OpenEvolve
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional
from openevolve.utils.git_utils import ensure_repo

from openevolve import OpenEvolve
from openevolve.config import Config, load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="OpenEvolve - Evolutionary coding agent")

    parser.add_argument("git_repo", help="Path to the git repository for commit-based evolution")

    parser.add_argument(
        "evaluation_file", help="Path to the evaluation file containing an 'evaluate' function"
    )

    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)

    parser.add_argument("--output", "-o", help="Output directory for results", default=None)

    parser.add_argument(
        "--iterations", "-i", help="Maximum number of iterations", type=int, default=None
    )

    parser.add_argument(
        "--target-score", "-t", help="Target score to reach", type=float, default=None
    )

    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory to resume from (e.g., openevolve_output/checkpoints/checkpoint_50)",
        default=None,
    )

    parser.add_argument("--api-base", help="Base URL for the LLM API", default=None)

    parser.add_argument("--primary-model", help="Primary LLM model name", default=None)

    parser.add_argument("--secondary-model", help="Secondary LLM model name", default=None)

    # --- Commit-based evolution parameters ---
    parser.add_argument("--root-commit", help="Baseline commit SHA for diff generation", default=None)
    parser.add_argument("--evolution-target", help="Natural language evolution target description", default=None)
    parser.add_argument("--similarity-threshold", type=float, help="Signature similarity threshold", default=None)

    # --- Long session management parameters ---
    parser.add_argument("--session-max-tokens", type=int, help="Maximum tokens per session", default=None)
    parser.add_argument("--session-compress-threshold", type=int, help="Token threshold to trigger session compression", default=None)
    parser.add_argument("--recent-history-tokens", type=int, help="Upper bound of recent history tokens to keep", default=None)
    parser.add_argument("--compression-model", help="Model name used for compressing session history", default=None)

    # --- Write tool model ---
    parser.add_argument("--write-tool-model", help="Model name for multi-file write operations", default=None)

    return parser.parse_args()


async def main_async() -> int:
    """
    Main asynchronous entry point

    Returns:
        Exit code
    """
    args = parse_args()

    # Check repo and evaluation file exist
    if not os.path.exists(args.git_repo) or not os.path.isdir(args.git_repo):
        print(f"Error: Repository path '{args.git_repo}' not found or not a directory")
        return 1
    try:
        ensure_repo(args.git_repo)
    except Exception:
        print(f"Error: Path '{args.git_repo}' is not a git repository")
        return 1

    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found")
        return 1

    # Create config object with command-line overrides
    config = None

    # Determine if any CLI overrides require an explicit Config object
    override_flags = [
        args.api_base,
        args.primary_model,
        args.secondary_model,
        args.root_commit,
        args.evolution_target,
        args.similarity_threshold,
        args.session_max_tokens,
        args.session_compress_threshold,
        args.recent_history_tokens,
        args.compression_model,
        args.write_tool_model,
    ]

    if any(flag is not None for flag in override_flags):
        # Load base config from file (if provided) or defaults
        config = load_config(args.config)

        # ---- LLM-related overrides ----
        if args.api_base:
            config.llm.update_model_params({"api_base": args.api_base}, overwrite=True)
            print(f"Using API base: {args.api_base}")

        if args.primary_model:
            if config.llm.models:
                config.llm.models[0].name = args.primary_model
            else:
                from openevolve.config import LLMModelConfig
                config.llm.models = [LLMModelConfig(name=args.primary_model, weight=1.0)]
            print(f"Using primary model: {args.primary_model}")

        if args.secondary_model:
            from openevolve.config import LLMModelConfig
            if len(config.llm.models) < 2:
                config.llm.models.append(LLMModelConfig(name=args.secondary_model, weight=0.5))
            else:
                config.llm.models[1].name = args.secondary_model
            print(f"Using secondary model: {args.secondary_model}")

        if args.write_tool_model:
            # Map name to a model config object if it matches one of the ensemble models
            from openevolve.config import LLMModelConfig
            chosen = None
            for m in config.llm.models:
                if m.name == args.write_tool_model:
                    chosen = m
                    break
            if chosen is None:
                chosen = LLMModelConfig(name=args.write_tool_model, weight=1.0)
            config.llm.write_tool_model = chosen
            print(f"Using write-tool model (object): {args.write_tool_model}")

        # ---- Database / commit evolution overrides ----
        if args.root_commit:
            config.database.root_commit = args.root_commit
            print(f"Using root commit: {config.database.root_commit}")

        if args.evolution_target is not None:
            config.database.evolution_target = args.evolution_target
            print("Set evolution target from CLI")

        if args.similarity_threshold is not None:
            config.database.signature_similarity_threshold = args.similarity_threshold
            print(f"Set similarity threshold: {config.database.signature_similarity_threshold}")

        # ---- Session management overrides ----
        if args.session_max_tokens is not None:
            config.prompt.session_max_tokens = args.session_max_tokens
            print(f"Session max tokens: {config.prompt.session_max_tokens}")

        if args.session_compress_threshold is not None:
            config.prompt.session_compress_threshold = args.session_compress_threshold
            print(
                f"Session compress threshold: {config.prompt.session_compress_threshold}"
            )

        if args.recent_history_tokens is not None:
            config.prompt.recent_history_tokens = args.recent_history_tokens
            print(f"Recent history tokens: {config.prompt.recent_history_tokens}")

        if args.compression_model is not None:
            # Prefer dedicated compression model object under llm
            from openevolve.config import LLMModelConfig
            chosen = None
            for m in config.llm.models:
                if m.name == args.compression_model:
                    chosen = m
                    break
            if chosen is None:
                chosen = LLMModelConfig(name=args.compression_model, weight=1.0)
            config.llm.compression_model = chosen
            print(f"Compression model (object): {args.compression_model}")

    # Initialize OpenEvolve
    try:
        openevolve = OpenEvolve(
            git_repo_path=args.git_repo,
            evaluation_file=args.evaluation_file,
            config=config,
            config_path=args.config if config is None else None,
            output_dir=args.output,
        )

        # Load from checkpoint if specified
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint directory '{args.checkpoint}' not found")
                return 1
            print(f"Loading checkpoint from {args.checkpoint}")
            openevolve.database.load(args.checkpoint)
            print(
                f"Checkpoint loaded successfully (iteration {openevolve.database.last_iteration})"
            )

        # Override log level if specified
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))

        # Run evolution
        best_program = await openevolve.run(
            iterations=args.iterations,
            target_score=args.target_score,
            checkpoint_path=args.checkpoint,
        )

        # Get the checkpoint path
        checkpoint_dir = os.path.join(openevolve.output_dir, "checkpoints")
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                os.path.join(checkpoint_dir, d)
                for d in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            if checkpoints:
                latest_checkpoint = sorted(
                    checkpoints, key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
                )[-1]

        print(f"\nEvolution complete!")
        if best_program is not None:
            print(f"Best program metrics:")
            for name, value in best_program.metrics.items():
                # Handle mixed types: format numbers as floats, others as strings
                if isinstance(value, (int, float)):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")
        else:
            print("No best program available.")

        if latest_checkpoint:
            print(f"\nLatest checkpoint saved at: {latest_checkpoint}")
            print(f"To resume, use: --checkpoint {latest_checkpoint}")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """
    Main entry point

    Returns:
        Exit code
    """
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
