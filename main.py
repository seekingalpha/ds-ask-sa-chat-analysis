"""Entry point for the Ask SA chat analysis jobs.

One task per process, so each SageMaker pipeline step gets a clean GPU:

    python main.py --task sentiment
    python main.py --task ner
    python main.py --task ner --dry-run --max-rows 50   # quick test, no writes

Exit code 0 = success (including "nothing new to process"), 1 = the task
failed; SageMaker marks the step as Failed on a non-zero exit code.
"""

import argparse
import importlib
import logging
import os
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Must be set before torch / transformers / tqdm are imported
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# progress bars only flood CloudWatch; transformers must not import TensorFlow
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Works both with the package installed (container) and from a plain checkout
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asksa_jobs.common import load_settings  # noqa: E402
from logger.logger_setup import setup_logger  # noqa: E402

logger = logging.getLogger("main")

TASKS = {
    "sentiment": "asksa_jobs.sentiment_job",
    "ner": "asksa_jobs.ner_job",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask SA chat analysis jobs")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run everything except writing to S3 / Redshift",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="process at most this many rows with the LLM (for tests)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="also write DEBUG logs to this file (default: console only)",
    )
    return parser.parse_args(argv)


def log_environment() -> None:
    """Versions and GPU, so every run's log says what it ran on."""
    logger.info(f"Python {platform.python_version()} on {platform.platform()}")
    try:
        import torch
        import transformers

        gpu = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "none"
        )
        logger.info(
            f"torch {torch.__version__} (CUDA {torch.version.cuda}), "
            f"transformers {transformers.__version__}, GPU: {gpu}"
        )
    except Exception as e:  # versions are informational only
        logger.warning(f"Could not read torch/transformers versions: {e}")


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logger(log_file=args.log_file)
    logger.info(
        f"Task: {args.task}, dry run: {args.dry_run}, max rows: {args.max_rows}"
    )

    start = time.time()
    try:
        log_environment()
        settings = load_settings(PROJECT_ROOT)
        job = importlib.import_module(TASKS[args.task])
        rows = job.run(settings, dry_run=args.dry_run, max_rows=args.max_rows)
    except Exception:
        logger.exception(
            f"Task {args.task} FAILED after {(time.time() - start) / 60:.1f} min"
        )
        return 1

    action = "prepared (dry run)" if args.dry_run else "loaded"
    logger.info(
        f"Task {args.task} finished in {(time.time() - start) / 60:.1f} min, "
        f"rows {action}: {rows}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
