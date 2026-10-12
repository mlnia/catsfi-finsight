import json

from finsight.eval.runner import main


def test_eval_gate_passes_on_fixture(tmp_path):
    code = main(["--dataset", "fixture", "--modes", "hybrid,sparse", "--prompts", "rag_answer@v1,rag_answer@v2",
                 "--out", str(tmp_path)])
    assert code == 0
    report = json.loads(next(tmp_path.glob("eval_fixture_*.json")).read_text())
    assert len(report["configs"]) == 4
    hybrid = report["configs"][0]["metrics"]
    assert hybrid["hit@5"] >= 0.9 and hybrid["abstention_accuracy"] >= 0.9
    assert report["injection"]["f1"] >= 0.9
    assert report["setup"]["injection_leaked_to_context"] is False
    assert (tmp_path / "latest_fixture.md").exists()


def test_eval_gate_fails_on_impossible_threshold(tmp_path):
    assert main(["--dataset", "fixture", "--modes", "hybrid", "--min-hit", "1.01", "--out", str(tmp_path)]) == 1
