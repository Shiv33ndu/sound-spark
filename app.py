import json, numpy as np, os
from src.orchestrator import Orchestrator

orc = Orchestrator()

sample = "tests/sample_audio/vocal_chop.wav"

out = orc.run(sample)

print(json.dumps(out, indent=2))