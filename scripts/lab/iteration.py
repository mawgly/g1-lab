"""Writes one iteration of the lab council into the journal («Совет лаборатории»): iteration.py <file.json>.

The JSON file (UTF-8): {"id": "iter-1", "title": "...", "question": "...",
                        "messages": [{"role": "owner|lead|critic|scout|analyst|watch|chronicler|engineer|reviewer", "text": "..."}],
                        "decision": "..." | null}
The same id replaces the earlier version (e.g. once the decision is taken)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim import journal  # noqa: E402
from humanoid_sim.core import DEFAULT_RUN  # noqa: E402
from humanoid_sim.lab import Laboratory  # noqa: E402

item = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
journal.add_iteration(Laboratory(DEFAULT_RUN).root, item['id'], item['title'], item['question'], item['messages'],
                      item.get('decision'))
print('ok', item['id'])
