def test_query_endpoint(client):
    response = client.post(
        "/query/",
        json={
            "query": "Why does system X fail under load?",
            "entities": ["system X"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "confidence" in body
    assert "uncertainty" in body
    assert "shock" in body
