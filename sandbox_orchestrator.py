"""
Orchestrator that spins up an Apify AI Sandbox with autoresearch code,
installs dependencies, prepares data, and exposes MCP/shell endpoints
for an AI agent to drive the research loop.

Usage:
    uv run sandbox_orchestrator.py
    uv run sandbox_orchestrator.py --time-budget 300 --num-shards 10

Requires APIFY_TOKEN environment variable.
"""

import os
import sys
import time
import argparse
import json

import requests
from apify_client import ApifyClient


SANDBOX_ACTOR_ID = "apify/ai-sandbox"

# Files to upload into the sandbox
FILES_TO_UPLOAD = ["prepare.py", "train.py", "program.md"]

# Python dependencies (excluding torch - installed separately via init script)
PYTHON_REQUIREMENTS = """\
numpy>=2.2.6
pandas>=2.3.3
pyarrow>=21.0.0
requests>=2.32.0
rustbpe>=0.1.0
tiktoken>=0.11.0
matplotlib>=3.10.8
"""

# Init script installs CPU-only torch (not available via requirements.txt --index-url)
INIT_SCRIPT = """\
#!/bin/bash
set -e
echo "Installing CPU-only PyTorch..."
/sandbox/py/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
echo "PyTorch CPU installed successfully."
"""


def wait_for_sandbox(base_url: str, timeout: int = 600) -> bool:
    """Wait for the sandbox to become healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/health", timeout=10)
            data = resp.json()
            status = data.get("status")
            if status == "healthy":
                return True
            print(f"  Sandbox status: {status} ({int(time.time() - start)}s elapsed)")
        except (requests.RequestException, json.JSONDecodeError):
            print(f"  Waiting for sandbox... ({int(time.time() - start)}s elapsed)")
        time.sleep(10)
    return False


def upload_file(base_url: str, local_path: str, remote_path: str) -> None:
    """Upload a file to the sandbox via /fs endpoint."""
    with open(local_path, "rb") as f:
        content = f.read()
    resp = requests.put(
        f"{base_url}/fs/{remote_path}",
        data=content,
        headers={"Content-Type": "application/octet-stream"},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    print(f"  Uploaded {local_path} -> /sandbox/{remote_path} ({result.get('size', '?')} bytes)")


def exec_command(base_url: str, command: str, timeout_secs: int = 600, cwd: str = "/sandbox/py") -> dict:
    """Execute a shell command in the sandbox."""
    resp = requests.post(
        f"{base_url}/exec",
        json={"command": command, "cwd": cwd, "timeoutSecs": timeout_secs},
        timeout=timeout_secs + 30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Launch autoresearch AI Sandbox on Apify")
    parser.add_argument("--time-budget", type=int, default=300, help="Training time budget in seconds (default: 300)")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of data shards to download (default: 10)")
    parser.add_argument("--memory-mb", type=int, default=8192, help="Sandbox memory in MB (default: 8192)")
    parser.add_argument("--idle-timeout", type=int, default=3600, help="Sandbox idle timeout in seconds (default: 3600)")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip data preparation step")
    parser.add_argument("--run-baseline", action="store_true", help="Run baseline training after setup")
    args = parser.parse_args()

    token = os.environ.get("APIFY_TOKEN")
    if not token:
        # Try reading from Apify CLI auth
        auth_path = os.path.expanduser("~/.apify/auth.json")
        if os.path.exists(auth_path):
            import json as _json
            with open(auth_path) as f:
                token = _json.load(f).get("token")
    if not token:
        print("Error: APIFY_TOKEN not set and no Apify CLI auth found.")
        print("Either set APIFY_TOKEN or run: apify login")
        sys.exit(1)

    # Check that source files exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in FILES_TO_UPLOAD:
        fpath = os.path.join(script_dir, fname)
        if not os.path.exists(fpath):
            print(f"Error: Required file not found: {fpath}")
            sys.exit(1)

    # Start the AI Sandbox
    print("Starting AI Sandbox on Apify...")
    client = ApifyClient(token)

    run_input = {
        "pythonRequirementsTxt": PYTHON_REQUIREMENTS,
        "initShellScript": INIT_SCRIPT,
        "idleTimeoutSeconds": args.idle_timeout,
    }

    run = client.actor(SANDBOX_ACTOR_ID).start(
        run_input=run_input,
        memory_mbytes=args.memory_mb,
        timeout_secs=0,  # infinite - let idle timeout manage lifecycle
    )

    run_id = run["id"]
    container_url = run.get("containerUrl", "")
    # The containerUrl from the API might need the protocol
    if container_url and not container_url.startswith("http"):
        base_url = f"https://{container_url}"
    else:
        base_url = container_url

    print(f"  Run ID: {run_id}")
    print(f"  Base URL: {base_url}")
    print(f"  Console: https://console.apify.com/actors/runs/{run_id}")

    # Wait for sandbox to be healthy
    print("\nWaiting for sandbox to initialize (installing dependencies)...")
    if not wait_for_sandbox(base_url, timeout=600):
        print("Error: Sandbox failed to become healthy within 10 minutes.")
        print("Check the run logs at the console URL above.")
        sys.exit(1)

    print("Sandbox is healthy!\n")

    # Upload autoresearch files
    print("Uploading autoresearch files...")
    for fname in FILES_TO_UPLOAD:
        fpath = os.path.join(script_dir, fname)
        upload_file(base_url, fpath, f"py/{fname}")

    # Run data preparation
    if not args.skip_prepare:
        print(f"\nRunning data preparation (downloading {args.num_shards} shards + training tokenizer)...")
        print("  This may take a few minutes...")
        result = exec_command(
            base_url,
            f"/sandbox/py/venv/bin/python prepare.py --num-shards {args.num_shards}",
            timeout_secs=600,
        )
        if result.get("exitCode") != 0:
            print(f"  FAILED (exit code {result.get('exitCode')}):")
            print(f"  stdout: {result.get('stdout', '')[:500]}")
            print(f"  stderr: {result.get('stderr', '')[:500]}")
            sys.exit(1)
        print("  Data preparation complete!")
        if result.get("stdout"):
            # Print last few lines of output
            lines = result["stdout"].strip().split("\n")
            for line in lines[-5:]:
                print(f"    {line}")

    # Optionally run baseline
    if args.run_baseline:
        print(f"\nRunning baseline training ({args.time_budget}s time budget)...")
        result = exec_command(
            base_url,
            f"/sandbox/py/venv/bin/python train.py",
            timeout_secs=args.time_budget + 120,
        )
        if result.get("exitCode") != 0:
            print(f"  FAILED (exit code {result.get('exitCode')}):")
            print(f"  stderr: {result.get('stderr', '')[:500]}")
        else:
            print("  Baseline training complete!")
            # Extract val_bpb from output
            for line in result.get("stdout", "").split("\n"):
                if "val_bpb" in line:
                    print(f"    {line.strip()}")

    # Print connection details
    print("\n" + "=" * 60)
    print("AUTORESEARCH SANDBOX READY")
    print("=" * 60)
    print(f"\nSandbox URL:  {base_url}")
    print(f"Shell:        {base_url}/shell/")
    print(f"MCP endpoint: {base_url}/mcp")
    print(f"Health:       {base_url}/health")
    print(f"Console:      https://console.apify.com/actors/runs/{run_id}")
    print(f"\nFiles are in: /sandbox/py/")
    print(f"Python venv:  /sandbox/py/venv/bin/python")
    print()
    print("Connect Claude Code via MCP:")
    print(f'  claude mcp add --transport http autoresearch-sandbox {base_url}/mcp')
    print()
    print("Or run training manually:")
    print(f'  curl -X POST {base_url}/exec \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"command": "/sandbox/py/venv/bin/python train.py", "cwd": "/sandbox/py", "timeoutSecs": 600}}\'')
    print()
    print("Modify train.py:")
    print(f'  curl -X PUT {base_url}/fs/py/train.py \\')
    print(f'    -H "Content-Type: text/plain" \\')
    print(f'    --data-binary @train.py')
    print()
    print(f"Idle timeout: {args.idle_timeout}s (sandbox auto-stops after inactivity)")
    print("=" * 60)


if __name__ == "__main__":
    main()
