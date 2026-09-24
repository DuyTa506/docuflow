from config.capacity import batched, capacity_profile


def test_batched_splits_group_a():
    stages = ["a", "b", "c", "d"]
    assert batched(stages, 2) == [["a", "b"], ["c", "d"]]
    assert batched(stages, 4) == [stages]
    assert batched(stages, 0) == [["a"], ["b"], ["c"], ["d"]]


def test_profile_reads_pipeline_cap():
    cap = capacity_profile()
    assert cap.max_digest_pipelines >= 1
    assert cap.max_extractions >= 1
    assert cap.digest_group_a_parallelism >= 1
