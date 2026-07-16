import json
from pathlib import Path

from pypdf import PdfReader


def test_report_has_required_sections() -> None:
    text = Path("progress_report.tex").read_text()
    for heading in [
        "Brief Project Description",
        "Notable Contribution",
        "Data Processing",
        "Baseline Model",
        "Primary Model",
    ]:
        assert heading in text


def test_report_numbers_come_from_saved_results() -> None:
    text = Path("progress_report.tex").read_text()
    results = json.loads(Path("artifacts/metrics/results.json").read_text())
    assert results["configuration"]["dry_run"] is False
    for model in ["momentum", "logistic_regression", "cnn_lstm"]:
        assert f'{results["models"][model]["accuracy"]:.3f}' in text
        assert f'{results["models"][model]["macro_f1"]:.3f}' in text
    assert str(results["model_parameter_count"]) in text


def test_report_includes_never_seen_data_plan_and_limitations() -> None:
    text = Path("progress_report.tex").read_text().lower()
    assert "after july 1, 2026" in text
    assert "not used for tuning" in text
    assert "did not outperform" in text
    assert "not trading advice" in text


def test_report_neutralizes_undefined_course_footer() -> None:
    text = Path("progress_report.tex").read_text()
    definition = r"\providecommand{\@trackname}{}"
    assert definition in text
    assert text.index(definition) < text.index(r"\usepackage{APS360}")


def test_readme_documents_local_and_colab_reproduction() -> None:
    text = Path("README.md").read_text()
    assert "python run_experiment.py" in text
    assert "colab.research.google.com" in text
    assert "progress_report.pdf" in text
    assert "48.0%" in text
    assert "44.4%" in text


def test_compiled_report_has_three_main_pages_plus_references() -> None:
    reader = PdfReader("progress_report.pdf")
    assert len(reader.pages) == 4
    assert "Feasibility and next steps" in (reader.pages[2].extract_text() or "")
    assert "References" in (reader.pages[3].extract_text() or "")
