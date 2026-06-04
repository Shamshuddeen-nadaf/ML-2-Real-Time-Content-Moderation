def test_toxic_comment(client):
  resp = client.post("/moderate",json={"text":"Idiot."})
  assert resp.status_code ==200
  data = resp.json()
  assert data['toxic'] is True
  assert data['probability'] > 0.5

def test_clean_comment(client):
    resp = client.post("/moderate", json={"text": "thanks for your help"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["toxic"] is False
    assert data["probability"] < 0.5


def test_empty_string(client):
    resp = client.post("/moderate", json={"text": ""})
    assert resp.status_code == 200
    assert "toxic" in resp.json()


def test_missing_field(client):
    resp = client.post("/moderate", json={})
    assert resp.status_code == 422


def test_response_shape(client):
    resp = client.post("/moderate", json={"text": "hello"})
    data = resp.json()
    assert set(data.keys()) == {"toxic", "confidence", "probability"}