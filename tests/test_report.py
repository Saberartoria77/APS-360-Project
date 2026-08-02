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


def test_report_uses_numeric_square_bracket_citations() -> None:
    text = Path("progress_report.tex").read_text()
    assert r"\setcitestyle{numbers,square}" in text


def test_report_reuses_shared_bibliography() -> None:
    text = Path("progress_report.tex").read_text()
    assert r"\bibliography{refs}" in text
    assert not Path("progress_refs.bib").exists()


def test_report_links_public_github_repository() -> None:
    text = Path("progress_report.tex").read_text()
    assert r"\href{https://github.com/Saberartoria77/APS-360-Project}{GitHub repository}" in text


def test_final_report_has_every_rubric_section() -> None:
    text = Path("final_report.tex").read_text()
    for heading in [
        "Introduction",
        "Related Work",
        "Data Processing",
        "Architecture",
        "Baseline Models",
        "Quantitative Results",
        "Qualitative Results",
        "New Data",
        "Discussion",
        "Ethical Considerations",
    ]:
        assert heading in text


def test_final_report_uses_genuine_saved_results() -> None:
    historical = json.loads(
        Path("artifacts/final/historical_results.json").read_text()
    )
    prospective = json.loads(
        Path("artifacts/final/prospective_results.json").read_text()
    )
    assert historical["configuration"]["dry_run"] is False
    assert historical["configuration"]["data_mode"] == "genuine"
    assert prospective["configuration"]["data_mode"] == "genuine"
    assert prospective["configuration"]["prospective_revealed"] is True
    text = Path("final_report.tex").read_text()
    metric = prospective["slices"]["overall"]["models"]["cnn_lstm"][
        "macro_f1"
    ]
    assert f"{metric:.3f}" in text

    historical_cnn = historical["global_evaluation"]["slices"]["overall"][
        "models"
    ]["cnn_lstm"]
    assert f'{historical_cnn["macro_f1"]["mean"]:.3f}' in text
    assert f'{historical_cnn["macro_f1"]["std"]:.3f}' in text
    prospective_cnn = prospective["slices"]["overall"]["models"]["cnn_lstm"]
    assert f'{prospective_cnn["accuracy"]:.3f}' in text

    assert sum(prospective["data"]["class_counts"]) == prospective["data"][
        "sample_count"
    ]
    assert sum(prospective["data"]["regime_counts"].values()) == prospective[
        "data"
    ]["sample_count"]
    assert f'{prospective["data"]["sample_count"]:,}' in text
    for count in prospective["data"]["class_counts"]:
        assert f"{count:,}" in text
    for count in prospective["data"]["regime_counts"].values():
        assert f"{count:,}" in text

    transfers = historical["cross_regime_evaluation"][
        "paired_cnn_transfer_macro_f1_changes"
    ]
    for transfer in transfers:
        assert f'{abs(transfer["macro_f1_change_mean"]):.3f}' in text
        assert f'{transfer["macro_f1_change_std"]:.3f}' in text


def test_final_report_data_example_and_architecture_match_code() -> None:
    text = Path("final_report.tex").read_text()
    normalized = " ".join(text.split())
    assert "2026-06-30 22:00 UTC" in text
    assert "58,639.99" in text
    assert "58,607.99" in text
    assert "38.365" in text
    assert "linear logit head" in text
    assert "Softmax is applied only at inference" in normalized
    assert "inverse-frequency" in text
    assert r"\textbf{0.481}" in text

    figure_source = Path("src/final_evaluation.py").read_text()
    assert "3-class logits" in figure_source
    assert "3-class softmax" not in figure_source


def test_readme_documents_final_evaluation_artifacts() -> None:
    text = Path("README.md").read_text()
    for required in [
        "run_final_experiment.py",
        "artifacts/final/",
        "tectonic final_report.tex",
        "final_report.pdf",
    ]:
        assert required in text
    assert "deferred to the final project" not in text.lower()


def test_final_report_has_four_main_pages_plus_references() -> None:
    reader = PdfReader("final_report.pdf")
    assert len(reader.pages) == 5
    assert "Discussion" in (reader.pages[3].extract_text() or "")
    assert "References" in (reader.pages[4].extract_text() or "")
