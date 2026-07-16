import json
from pathlib import Path

import nbformat

from run_experiment import run_experiment


def test_experiment_dry_run_writes_required_artifacts(tmp_path: Path) -> None:
    result = run_experiment(output_dir=tmp_path, dry_run=True, epochs=1)
    required = [
        "metrics/results.json",
        "metrics/prediction_examples.csv",
        "figures/data_summary.png",
        "figures/confusion_momentum.png",
        "figures/confusion_logistic_regression.png",
        "figures/confusion_cnn_lstm.png",
        "figures/learning_curves.png",
        "figures/model_diagram.png",
    ]
    assert all((tmp_path / name).exists() for name in required)
    assert result["configuration"]["dry_run"] is True
    assert set(result["models"]) == {"momentum", "logistic_regression", "cnn_lstm"}
    assert result["data"]["train_samples"] > 0
    assert json.loads((tmp_path / "metrics/results.json").read_text())["model_parameter_count"] > 0


def test_progress_notebook_is_valid_and_uses_shared_pipeline() -> None:
    notebook = nbformat.read("notebooks/progress_report.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "run_experiment" in source
    assert "Academic integrity" in source
    assert "Student verification checklist" in source
