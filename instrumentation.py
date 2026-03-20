import json
import uuid
from datetime import datetime

LOG_FILE = "data/processed/agent_logs.jsonl"

def log_agent_call(agent, user_a_id, user_b_id, input_portrait_a,
                   input_portrait_b, input_moments, reasoning_trace, verdict):
    entry = {
        "log_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "agent": agent,
        "user_a_id": user_a_id,
        "user_b_id": user_b_id,
        "input_portrait_a": input_portrait_a,
        "input_portrait_b": input_portrait_b,
        "input_moments": input_moments,
        "reasoning_trace": reasoning_trace,
        "verdict": verdict,
        "user_outcome": None
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[instrumentation] Logging failed: {e}")