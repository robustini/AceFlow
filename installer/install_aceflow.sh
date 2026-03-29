#!/usr/bin/env bash
set -euo pipefail

repo_archive_url="https://github.com/robustini/AceFlow/archive/refs/heads/main.zip"
soundfont_url="https://musical-artifacts.com/artifacts/3677/LiteGM_v1.03.sf2"
soundfont_filename="LiteGM_v1.03.sf2"
working_root="$(pwd)"
acestep_dir="$working_root/acestep"

if [ ! -d "$acestep_dir" ]; then
  echo "Current directory does not look like an ACE-Step root. Missing folder: $acestep_dir" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required but was not found in PATH." >&2
  exit 1
fi

tmp_root="$(mktemp -d)"
zip_path="$tmp_root/aceflow_main.zip"
extract_dir="$tmp_root/extract"

cleanup() {
  rm -rf "$tmp_root"
}
trap cleanup EXIT

mkdir -p "$extract_dir"

echo "Downloading AceFlow repository archive..."
curl -fL "$repo_archive_url" -o "$zip_path"

echo "Extracting archive..."
unzip -q "$zip_path" -d "$extract_dir"

repo_root="$extract_dir/AceFlow-main"
if [ ! -d "$repo_root" ]; then
  repo_root="$(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
fi

if [ -z "${repo_root:-}" ] || [ ! -d "$repo_root" ]; then
  echo "Unable to locate extracted repository root." >&2
  exit 1
fi

source_ui="$repo_root/acestep/ui/aceflow"
source_bat="$repo_root/start_aceflow_ui.bat"
source_sh="$repo_root/start_aceflow_ui.sh"

if [ ! -d "$source_ui" ]; then
  echo "Missing source folder in archive: acestep/ui/aceflow" >&2
  exit 1
fi
if [ ! -f "$source_bat" ]; then
  echo "Missing source file in archive: start_aceflow_ui.bat" >&2
  exit 1
fi
if [ ! -f "$source_sh" ]; then
  echo "Missing source file in archive: start_aceflow_ui.sh" >&2
  exit 1
fi

target_ui_parent="$working_root/acestep/ui"
target_ui="$target_ui_parent/aceflow"
target_bat="$working_root/start_aceflow_ui.bat"
target_sh="$working_root/start_aceflow_ui.sh"
target_soundfonts_dir="$target_ui/soundfonts"
target_soundfont="$target_soundfonts_dir/$soundfont_filename"

mkdir -p "$target_ui_parent"

if [ -d "$target_ui" ]; then
  echo "Removing previous acestep/ui/aceflow..."
  rm -rf "$target_ui"
fi

echo "Installing AceFlow files..."
cp -R "$source_ui" "$target_ui_parent/"
cp "$source_bat" "$target_bat"
cp "$source_sh" "$target_sh"
chmod +x "$target_sh"
mkdir -p "$target_soundfonts_dir"

echo "Downloading default SoundFont into acestep/ui/aceflow/soundfonts/..."
if curl -fL "$soundfont_url" -o "$target_soundfont"; then
  sf_header_hex="$(od -An -t x1 -N 12 "$target_soundfont" | tr -d '[:space:]')"
  sf_prefix="${sf_header_hex:0:8}"
  sf_kind="${sf_header_hex:16:8}"
  if [ "$sf_prefix" = "52494646" ] && [ "$sf_kind" = "7366626b" ]; then
    echo "Default SoundFont installed: $target_soundfont"
  else
    echo "Warning: downloaded file does not look like a valid .sf2. Removing it and keeping AceFlow installed." >&2
    rm -f "$target_soundfont"
  fi
else
  echo "Warning: unable to download the default SoundFont. AceFlow was installed anyway and can still use the internal chord synth fallback." >&2
  rm -f "$target_soundfont" 2>/dev/null || true
fi

echo
echo "AceFlow installation completed."
echo "To run AceFlow, launch start_aceflow_ui.bat or start_aceflow_ui.sh from the ACE-Step root."
if [ -f "$target_soundfont" ]; then
  echo "Installed default SoundFont: $target_soundfont"
else
  echo "Default SoundFont is not present. AceFlow can still render chord references with the internal synth, or you can place a valid .sf2 manually in $target_soundfonts_dir"
fi
