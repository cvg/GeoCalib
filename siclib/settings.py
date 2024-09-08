from pathlib import Path

# flake8: noqa
# mypy: ignore-errors
try:
    from settings import DATA_PATH, EVAL_PATH, TRAINING_PATH
except ModuleNotFoundError:
    # @TODO: Add a way to patch paths
    root = Path(__file__).parent.parent  # top-level directory
    DATA_PATH = root / "data/"  # datasets and pretrained weights
    TRAINING_PATH = root / "outputs/training/"  # training checkpoints
    EVAL_PATH = root / "outputs/results/"  # evaluation results
