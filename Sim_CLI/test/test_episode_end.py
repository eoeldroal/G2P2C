from fastapi.testclient import TestClient
from Sim_CLI.main import app, app_state
from pathlib import Path
import json

client = TestClient(app)

def test_episode_end_creates_file(tmp_path):
    # Ensure buffer has one experience
    app_state["experience_buffer"] = [{"state": {}, "action": 0.0, "reward": 0.0, "next_state": None}]
    app_state["current_episode"] = 99
    resp = client.post("/episode_end")
    assert resp.status_code == 200
    data = resp.json()
    assert data["episode"] == 99
    exp_path = Path("results/dmms_experience/episode_99.json")
    assert exp_path.is_file()
    with open(exp_path) as f:
        saved = json.load(f)
    assert isinstance(saved, list)
