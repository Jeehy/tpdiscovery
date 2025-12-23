# targets/playbook.py
import os
import json
import time
import uuid
from typing import List, Dict

BASE_DIR = os.path.dirname(__file__)
PLAYBOOK_PATH = os.path.join(BASE_DIR, "hcc_playbook.json")

class Playbook:
    def __init__(self):
        self.strategies = self._load_strategies()

    def _load_strategies(self) -> List[Dict]:
        if os.path.exists(PLAYBOOK_PATH):
            try:
                with open(PLAYBOOK_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data.get("strategies", [])
            except Exception:
                return []
        return []

    def save(self):
        data = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "strategies": self.strategies
        }
        with open(PLAYBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_strategy(self, trace_data: Dict):
        import hashlib
        fingerprint = f"{trace_data.get('task')}-{trace_data.get('status')}-{str(trace_data.get('steps_summary'))}"
        fingerprint_hash = hashlib.md5(fingerprint.encode()).hexdigest()

        for s in self.strategies:
            if s.get("fingerprint") == fingerprint_hash:
                return

        entry = {
            "id": str(uuid.uuid4())[:8],
            "type": "execution_trace",
            "fingerprint": fingerprint_hash,
            "data": trace_data,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.strategies.append(entry)
        self.save()