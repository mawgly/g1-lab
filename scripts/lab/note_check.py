"""Laboratory.annotate from a file: note_check.py <branch> <verdict or -> <note file>."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN  # noqa: E402
from humanoid_sim.lab import Laboratory  # noqa: E402

branch, verdict, note = sys.argv[1], sys.argv[2], Path(sys.argv[3]).read_text(encoding='utf-8-sig').strip()
Laboratory(DEFAULT_RUN).annotate(branch, verdict=None if verdict == '-' else verdict, note=note)
print('ok')
