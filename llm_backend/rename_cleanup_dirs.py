"""
Clean up polluted directory names
Rename *_全网最全51cto课程+v：_51cto_download to clean names
"""

import os
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Project root directory
PROJECT_ROOT = Path(__file__).parent
POLLUTION_SUFFIX = "_全网最全51cto课程+v：_51cto_download"

# Statistics
renamed_dirs = []
skipped_dirs = []
errors = []

def rename_directory(dir_path: Path) -> bool:
    """
    Rename a single polluted directory

    Args:
        dir_path: Directory path

    Returns:
        True if renamed successfully, False if skipped
    """
    dir_name = dir_path.name

    # Check if it's a polluted directory
    if POLLUTION_SUFFIX not in dir_name:
        return False

    # Get new directory name (remove pollution suffix)
    # Handle nested pollution like: dir_suffix_suffix
    new_name = dir_name.replace(POLLUTION_SUFFIX, "")
    new_path = dir_path.parent / new_name

    # Skip if target directory already exists
    if new_path.exists() and new_path != dir_path:
        print(f"  [SKIP] Target exists: {new_path}")
        skipped_dirs.append(str(dir_path))
        return False

    # Rename
    try:
        dir_path.rename(new_path)
        renamed_dirs.append(str(dir_path))
        print(f"  [RENAME DIR] {dir_name} -> {new_name}")
        return True
    except Exception as e:
        print(f"  [ERROR] {dir_name}: {e}")
        errors.append((str(dir_path), str(e)))
        return False

def process_directory(root_dir: Path, depth: int = 0):
    """Recursively process directory"""
    indent = "  " * depth
    print(f"\n{indent}[DEPTH {depth}] Processing: {root_dir.relative_to(PROJECT_ROOT)}")

    items_to_process = []

    # First, collect all items to process
    for item in root_dir.iterdir():
        if POLLUTION_SUFFIX in item.name:
            items_to_process.append(item)

    # Process polluted directories (bottom-up by sorting)
    for item in sorted(items_to_process, key=lambda x: len(str(x)), reverse=True):
        rename_directory(item)

    # Recursively process subdirectories
    for item in root_dir.iterdir():
        if item.is_dir():
            process_directory(item, depth + 1)

def main():
    print("=" * 60)
    print("Starting cleanup of polluted directory names")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Pollution suffix: {POLLUTION_SUFFIX}")
    print("=" * 60)

    # Recursively process all directories
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir():
            process_directory(item)

    # Print statistics
    print("\n" + "=" * 60)
    print("Rename completed!")
    print(f"  Renamed directories: {len(renamed_dirs)}")
    print(f"  Skipped directories: {len(skipped_dirs)}")
    print(f"  Error directories: {len(errors)}")
    print("=" * 60)

    if errors:
        print("\nError details:")
        for old_path, error in errors:
            print(f"  {old_path}: {error}")

    # Save log
    log_path = PROJECT_ROOT / "rename_cleanup_dirs_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Directory Rename Log ===\n\n")
        f.write(f"Renamed directories: {len(renamed_dirs)}\n")
        f.write(f"Skipped directories: {len(skipped_dirs)}\n")
        f.write(f"Error directories: {len(errors)}\n\n")

        if renamed_dirs:
            f.write("--- Renamed directories ---\n")
            for path in renamed_dirs:
                f.write(f"  {path}\n")

        if skipped_dirs:
            f.write("\n--- Skipped directories ---\n")
            for path in skipped_dirs:
                f.write(f"  {path}\n")

        if errors:
            f.write("\n--- Error directories ---\n")
            for old_path, error in errors:
                f.write(f"  {old_path}: {error}\n")

    print(f"\nLog saved to: {log_path}")

if __name__ == "__main__":
    main()
