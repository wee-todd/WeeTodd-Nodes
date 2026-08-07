import mlx.core as mx

from minimax_h3_mlx.algorithm_search.benchmark import benchmark_candidate
from minimax_h3_mlx.algorithm_search.results import ExperimentStore
from minimax_h3_mlx.algorithm_search.schema import AlgorithmClass


def test_experiment_store_appends_jsonl(tmp_path):
    result = benchmark_candidate(
        lambda x: x,
        lambda x: x,
        (mx.ones((1,)),),
        candidate_id="identity",
        operator="test",
        algorithm_class=AlgorithmClass.EXACT,
        warmups=0,
        repetitions=1,
    )
    store = ExperimentStore(tmp_path / "results.jsonl")
    store.append(result, context={"git_commit": "test"})
    store.append(result)
    records = store.read()
    assert len(records) == 2
    assert records[0]["context"]["git_commit"] == "test"
    assert records[0]["algorithm_class"] == "exact"
    assert "recorded_at_utc" in records[0]["context"]
    assert records[1]["context"]["git_commit"]
