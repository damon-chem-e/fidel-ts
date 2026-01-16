# Glow Custom Theme Setup Instructions

This document provides instructions for setting up custom glow themes on Linux and Windows.

## Theme Files

Two custom themes are available in the `tui/` directory:
- **`dark-pro.json`** - Professional grayscale theme with muted colors
- **`matrix-wide.json`** - Bright green matrix-inspired theme

Both themes are configured for 140-character width rendering.

## Linux Setup

### 1. Install Theme Files

Copy the theme files to glow's config directory:

```bash
# For standard glow installation
mkdir -p ~/.config/glow
cp tui/dark-pro.json ~/.config/glow/
cp tui/matrix-wide.json ~/.config/glow/

# For snap installation
mkdir -p ~/snap/glow/current/.config/glow
cp tui/dark-pro.json ~/snap/glow/current/.config/glow/
cp tui/matrix-wide.json ~/snap/glow/current/.config/glow/
```

### 2. Configure Default Theme (Optional)

Edit `~/.config/glow/glow.yml` (or `~/snap/glow/current/.config/glow/glow.yml` for snap):

```yaml
# style name or JSON path (default "auto")
style: "/home/YOUR_USERNAME/.config/glow/dark-pro.json"
# OR for snap:
# style: "/home/YOUR_USERNAME/snap/glow/current/.config/glow/dark-pro.json"

# word-wrap at width
width: 140
```

### 3. Create Convenient Aliases

Add to `~/.bash_aliases` (or `~/.bashrc` if `.bash_aliases` doesn't exist):

```bash
# Glow custom themes
alias glow-matrix='glow -s ~/.config/glow/matrix-wide.json'
alias glow-pro='glow -s ~/.config/glow/dark-pro.json'
```

Then reload:
```bash
source ~/.bash_aliases
```

### 4. Usage

```bash
# Using aliases
glow-pro document.md
glow-matrix document.md

# Using -s flag directly
glow -s ~/.config/glow/dark-pro.json document.md
glow -s ~/.config/glow/matrix-wide.json -p document.md  # with pager
```

## Windows Setup (PowerShell)

### 1. Install Theme Files

Copy the theme files to glow's config directory:

```powershell
# Create config directory
New-Item -ItemType Directory -Force -Path "$env:APPDATA\glow"

# Copy theme files
Copy-Item "tui\dark-pro.json" "$env:APPDATA\glow\"
Copy-Item "tui\matrix-wide.json" "$env:APPDATA\glow\"
```

### 2. Configure Default Theme (Optional)

Create or edit `$env:APPDATA\glow\glow.yml`:

```yaml
# style name or JSON path (default "auto")
style: "C:\\Users\\YOUR_USERNAME\\AppData\\Roaming\\glow\\dark-pro.json"

# word-wrap at width
width: 140
```

### 3. Create PowerShell Functions (Aliases)

Edit your PowerShell profile:

```powershell
# Open profile for editing
notepad $PROFILE
```

Add these functions:

```powershell
# Glow custom themes
function glow-matrix { glow -s "$env:APPDATA\glow\matrix-wide.json" @args }
function glow-pro { glow -s "$env:APPDATA\glow\dark-pro.json" @args }
```

Reload your profile:
```powershell
. $PROFILE
```

### 4. Usage

```powershell
# Using functions
glow-pro document.md
glow-matrix document.md

# Using -s flag directly
glow -s "$env:APPDATA\glow\dark-pro.json" document.md
glow -s "$env:APPDATA\glow\matrix-wide.json" -p document.md  # with pager
```

## Theme Characteristics

### Dark-Pro Theme
- Professional grayscale palette
- Main text: Light gray
- Headers: White to light gray gradient
- Links: Muted blue
- Code: Tan/beige on dark gray
- Best for: Long reading sessions, professional documents

### Matrix-Wide Theme
- Bright matrix green palette
- Main text: Bright green
- Headers: Light green
- Links: Cyan-green
- Code: Green variants on dark gray
- Best for: Terminal aesthetic, high contrast

## Troubleshooting

### Theme Not Loading
- Verify the file path in `glow.yml` is absolute and correct
- Check file permissions (should be readable)
- Try using `-s` flag with full path to test

### Width Not Applied
- Width setting must be in `glow.yml`, not in theme JSON
- Ensure `glow.yml` is in the correct config directory

### Colors Look Wrong
- Ensure your terminal supports 256 colors
- Try different theme (dark-pro vs matrix-wide)
- Check terminal color scheme compatibility

## Additional Resources

- Glow documentation: https://github.com/charmbracelet/glow
- Glamour (underlying renderer): https://github.com/charmbracelet/glamour
- Create custom themes: See existing theme JSON files as templates
