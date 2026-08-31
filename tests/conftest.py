from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Unit tests use tiny tensors; one intra-op thread avoids severe oversubscription
# on large login-node CPU allocations and does not change numerical semantics.
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass
