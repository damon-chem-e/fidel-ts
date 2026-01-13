# Claude Code Instructions

## Remote Environment: Canada RunPod Instance

The local Windows environment doesn't have the required dependencies. Use the remote RunPod instance for Python commands and testing.

### Instance Details
- **Name**: canada pod
- **Dataset available**: canada photovoltaics data
- **Workspace**: `/workspace/fidel-ts`

### SSH Connection
```bash
ssh 708s8s9tby38jc-64411b16@ssh.runpod.io -i ~/.ssh/id_ed25519
```

## Tmux Sessions

Use tmux for persistent sessions on the RunPod instance. All Claude sessions should be prefixed with `claude-` to differentiate from user sessions.

### Creating/Attaching to a Session
```bash
# Create a new session
tmux new-session -s claude-main

# Attach to existing session
tmux attach-session -t claude-main

# List sessions
tmux list-sessions
```

### Session Naming Convention
- `claude-main` - Primary testing session
- `claude-build` - For build/compile operations
- `claude-test` - For running tests

### Saving Session Logs
Save session logs frequently to `/workspace/claude-sessions/`. Run this tmux command (Ctrl+B, then `:`, then enter):

```
capture-pane -pS - | tee /workspace/claude-sessions/claude-main.log
```

Or save the entire scrollback buffer:
```
capture-pane -pS -32768 > /workspace/claude-sessions/claude-main.log
```

**Note**: Create the log directory if it doesn't exist:
```bash
mkdir -p /workspace/claude-sessions
```

## Workflow: Testing Code Changes

**IMPORTANT**: Before testing changes on RunPod, sync the code:

1. **Local**: Commit and push changes
   ```bash
   git add -A && git commit -m "description" && git push
   ```

2. **RunPod**: Fetch and pull changes
   ```bash
   cd /workspace/fidel-ts
   git fetch && git pull
   ```

3. **RunPod**: Activate venv and test
   ```bash
   source .venv/bin/activate
   python -m cli.tensor_cache generate --help
   ```

## Quick Reference

```bash
# One-liner to connect and activate venv
ssh 708s8s9tby38jc-64411b16@ssh.runpod.io -i ~/.ssh/id_ed25519 -t "cd /workspace/fidel-ts && tmux attach -t claude-main || tmux new -s claude-main"

# Inside tmux session
cd /workspace/fidel-ts
source .venv/bin/activate
git fetch && git pull

# Save log before exiting
# Press Ctrl+B, then : , then type:
capture-pane -pS -32768 > /workspace/claude-sessions/claude-main.log
```
