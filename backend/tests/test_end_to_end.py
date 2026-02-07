def test_full_migraph_pipeline_end_to_end(client, memory):
    """
    End-to-end test covering:
    - API layer
    - Shock detection
    - Graph mutation
    - Agent judgment
    - Answer synthesis
    - Episodic memory write
    """

    request_payload = {
        "query": "Why do distributed systems fail under partial network partitions?",
        "entities": ["distributed systems", "network partitions"],
    }

    response = client.post("/query/", json=request_payload)

    assert response.status_code == 200

    body = response.json()

    # ---------------- Core outputs ----------------

    assert "answer" in body
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0

    assert "confidence" in body
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0

    # ---------------- Uncertainty ----------------

    assert "uncertainty" in body
    uncertainty = body["uncertainty"]

    assert "stability" in uncertainty
    assert "entropy" in uncertainty

    assert isinstance(uncertainty["entropy"], float)
    assert uncertainty["entropy"] >= 0.0

    # ---------------- Shock ----------------

    assert "shock" in body
    shock = body["shock"]

    assert "overall" in shock
    assert isinstance(shock["overall"], float)
    assert shock["overall"] >= 0.0

    assert "components" in shock
    assert isinstance(shock["components"], dict)

    # ---------------- Diagnostics ----------------

    diagnostics = body["diagnostics"]
    assert "judge" in diagnostics
    assert "episode_id" in diagnostics

    judge = diagnostics["judge"]

    assert "confidence" in judge
    assert "uncertainty" in judge
    assert "dominant_agents" in judge

    # ---------------- Memory ----------------

    episodes = memory.all()

    assert len(episodes) >= 1

    last_episode = episodes[-1]

    assert last_episode.query == request_payload["query"]
    assert last_episode.entities == request_payload["entities"]
    assert last_episode.answer == body["answer"]
    assert abs(last_episode.confidence - body["confidence"]) < 1e-6

    # ---------------- Epistemic sanity checks ----------------

    # High shock should not produce high confidence
    if shock["overall"] > 0.7:
        assert body["confidence"] < 0.9
