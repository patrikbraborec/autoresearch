---
name: research
description: Launch an autoresearch sandbox on Apify and run the autonomous research loop
disable-model-invocation: true
argument-hint: [--run-baseline] [--num-shards N] [--memory-mb N]
---

# Autoresearch Sandbox Launcher

Launch an AI Sandbox on Apify for autonomous LLM training research.

## Steps

1. Run the sandbox orchestrator to spin up a remote sandbox:

```bash
uv run sandbox_orchestrator.py $ARGUMENTS
```

2. Once the sandbox is ready, the script will print connection details including the MCP endpoint URL and shell URL.

3. Tell the user:
   - The sandbox URLs (shell, MCP, console)
   - How to connect Claude Code via MCP: `claude mcp add --transport http autoresearch-sandbox <MCP_URL>`
   - That the sandbox has `prepare.py`, `train.py`, and `program.md` uploaded
   - They can open the shell in their browser to manually inspect the sandbox
   - The sandbox will auto-stop after the idle timeout (default: 1 hour)

4. If `--run-baseline` was passed, report the baseline val_bpb result.

5. Ask the user if they want you to connect to the sandbox via MCP and start the autonomous research loop (following `program.md`).
