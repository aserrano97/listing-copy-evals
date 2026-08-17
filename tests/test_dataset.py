from listing_gen.evals.dataset import load_dataset


def test_dataset_loads_all_fixtures_with_metadata() -> None:
    dataset = load_dataset()
    assert len(dataset) >= 6
    by_id = {s.metadata["property"]["property_id"]: s for s in dataset}
    assert by_id[104].metadata["forbidden_claims"], "adversarial fixture must carry forbidden claims"
    assert by_id[101].metadata["forbidden_claims"] == []
    assert "Villa Mar Blava" in by_id[101].input
