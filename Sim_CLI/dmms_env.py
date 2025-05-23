import subprocess
from pathlib import Path
from typing import Any, Tuple, Dict

import gym
import numpy as np
import httpx


class DmmsEnv(gym.Env):
    """Gym 호환 환경 래퍼.

    DMMS.R 프로세스를 관리하고 FastAPI 서버와 통신한다.
    실제 시뮬레이터와 API 동작 시나리오에 맞춰 수정이 필요하다.
    """

    metadata = {"render.modes": []}

    def __init__(self, api_url: str, exe: Path, cfg: Path, log_path: Path, results_root: Path) -> None:
        super().__init__()
        self.api_url = api_url.rstrip("/")
        self.exe = Path(exe)
        self.cfg = Path(cfg)
        self.log_path = Path(log_path)
        self.results_root = Path(results_root)
        self.proc: subprocess.Popen | None = None
        self.episode = 0
        self.results_root.mkdir(parents=True, exist_ok=True)

    def _start_process(self, results_dir: Path) -> None:
        cmd = [str(self.exe), str(self.cfg), str(self.log_path), str(results_dir)]
        self.proc = subprocess.Popen(cmd)

    def _terminate_process(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    def reset(self) -> Any:
        """시뮬레이터를 새로 시작하고 초기 상태를 반환한다."""
        self.close()
        self.episode += 1
        self.results_dir = self.results_root / f"episode_{self.episode}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._start_process(self.results_dir)
        # 초기 상태는 외부 플러그인을 통해 서버로 전달되므로 이곳에서는 None 반환
        return None

    def step(self, action: float) -> Tuple[Any, float, bool, Dict]:
        """FastAPI 서버의 /env_step을 호출한다."""
        payload = {"action": action}
        try:
            resp = httpx.post(f"{self.api_url}/env_step", json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None, np.nan, True, {}

        obs = data.get("next_state")
        reward = float(data.get("reward", 0.0))
        done = bool(data.get("done", False))
        return obs, reward, done, {}

    def close(self) -> None:
        self._terminate_process()

