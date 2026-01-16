# TUI Themes for Glow Markdown Renderer

This directory contains custom Glow themes for rendering markdown documents in the terminal.

## Themes

- **`dark-pro.json`** - Professional grayscale theme with muted, easy-on-the-eyes colors
- **`matrix-wide.json`** - Bright green matrix-inspired terminal aesthetic

Both themes are configured for 140-character width rendering with clean box-drawing headers and horizontal rules.

## Setup Instructions

See `.claude/glow-theme-setup.md` for complete installation and configuration instructions for:
- Linux (standard and snap installations)
- Windows (PowerShell)

## Quick Start (Linux)

```bash
# Copy themes to glow config directory
mkdir -p ~/.config/glow
cp dark-pro.json matrix-wide.json ~/.config/glow/

# Use with -s flag
glow -s ~/.config/glow/dark-pro.json ../context/experiment_tracking/experiment_tracker.md
```

## Quick Start (Windows PowerShell)

```powershell
# Copy themes to glow config directory
New-Item -ItemType Directory -Force -Path "$env:APPDATA\glow"
Copy-Item dark-pro.json,matrix-wide.json "$env:APPDATA\glow\"

# Use with -s flag
glow -s "$env:APPDATA\glow\dark-pro.json" ..\context\experiment_tracking\experiment_tracker.md
```

## Features

- 140-character width for comfortable reading
- Box-drawing characters for headers and sections
- ASCII status indicators (no emoji)
- Optimized for terminal viewing
- Clean, professional table rendering
