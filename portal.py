"""
Local web dashboard for the paper-trading portfolio.

Run it from this project directory:

    pip install -r requirements.txt
    python portal.py

Then open http://localhost:5000 in your browser. It shows the current
portfolio (performance chart, holdings, recent trades) read straight from
the files in this folder, and has two buttons:

  - "Refresh market data" -- re-fetches prices/fundamentals and marks the
    portfolio to market (adds a no-trade snapshot to the performance
    history). No trades are proposed or executed.
  - "Run full pipeline" -- does the same, then runs decide_trades.py (a
    simple, transparent scoring rule -- see that file's docstring) to
    propose trades, and executes whatever clears the risk checks in
    apply_trades.py. This is the automated stand-in for the "read
    today_data.json and decide" step in TRADING_INSTRUCTIONS.md, which
    normally takes a person or an LLM session reading it.

This file doesn't reimplement any trading logic -- it just runs the
existing scripts (prepare_data.py, decide_trades.py, apply_trades.py,
generate_report.py) as real subprocesses, the same way you'd run them by
hand, and serves the results to the browser. Everything stays on your own
machine; nothing here talks to GitHub or any external service.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "portal_static"
PYTHON = sys.executable

app = Flask(__name__)

job_lock = threading.Lock()
job_state = {
    "status": "idle",   # idle | running | done | error
    "mode": None,
    "steps": [],
    "error": None,
    "started_at": None,
    "finished_at": None,
}


def _run_step(name, args, timeout=600):
    step = {"name": name, "status": "running", "output": ""}
    job_state["steps"].append(step)
    result = subprocess.run(
        [PYTHON, *args], cwd=BASE_DIR, capture_output=True, text=True, timeout=timeout,
    )
    step["output"] = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        step["status"] = "failed"
        raise RuntimeError(f"{name} exited with code {result.returncode} -- see its output above.")
    step["status"] = "done"
    return result.stdout


def _candidate_count():
    data_file = BASE_DIR / "today_data.json"
    if not data_file.exists():
        return 0
    with open(data_file, "r", encoding="utf-8") as f:
        return len(json.load(f).get("candidates", []))


def _run_pipeline(mode):
    with job_lock:
        job_state.update(status="running", mode=mode, steps=[], error=None,
                          started_at=time.time(), finished_at=None)
    try:
        _run_step("prepare_data.py", ["prepare_data.py"])

        if _candidate_count() == 0:
            raise RuntimeError(
                "prepare_data.py returned 0 candidates -- stopping rather than "
                "deciding on bad data (same rule TRADING_INSTRUCTIONS.md gives "
                "for a manual run)."
            )

        if mode == "refresh":
            with open(BASE_DIR / "decision.json", "w", encoding="utf-8") as f:
                json.dump({"trades": [], "commentary": "Manual data refresh -- no trades."}, f)
        else:
            _run_step("decide_trades.py", ["decide_trades.py"])

        _run_step("apply_trades.py", ["apply_trades.py", "decision.json"])
        _run_step("generate_report.py", ["generate_report.py"])

        job_state["status"] = "done"
    except Exception as e:
        job_state["status"] = "error"
        job_state["error"] = str(e)
    finally:
        job_state["finished_at"] = time.time()


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/performance_chart.svg")
def chart():
    return send_from_directory(BASE_DIR, "performance_chart.svg")


@app.route("/api/report")
def api_report():
    # Imported lazily so the server still starts even before the first
    # portfolio.json / daily_history exists.
    from reporting import build_report_data
    data = build_report_data()
    if data is None:
        return jsonify({"error": "No trade history yet -- run the pipeline at least once."}), 404
    return jsonify(data)


@app.route("/api/run/<mode>", methods=["POST"])
def api_run(mode):
    if mode not in ("refresh", "full"):
        return jsonify({"error": "unknown mode"}), 400
    with job_lock:
        if job_state["status"] == "running":
            return jsonify({"error": "a run is already in progress"}), 409
    thread = threading.Thread(target=_run_pipeline, args=(mode,), daemon=True)
    thread.start()
    return jsonify({"started": True})


@app.route("/api/job")
def api_job():
    return jsonify(job_state)


if __name__ == "__main__":
    print("Portal running at http://localhost:5000  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False)
