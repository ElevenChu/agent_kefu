"""
Clean up polluted file names
Rename *_全网最全51cto课程+v：_51cto_download.py to *.py
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
POLLUTION_SUFFIX = "_全网最全51cto课程+v：_51cto_download.py"

# Statistics
renamed_files = []
skipped_files = []
errors = []

def rename_file(file_path: Path) -> bool:
    """
    Rename a single polluted file

    Args:
        file_path: File path

    Returns:
        True if renamed successfully, False if skipped
    """
    # Check if it's a polluted file
    if not file_path.name.endswith(POLLUTION_SUFFIX):
        return False

    # Get new file name (remove pollution suffix)
    original_name = file_path.name[:-len(POLLUTION_SUFFIX)]
    new_name = original_name + ".py"
    new_path = file_path.parent / new_name

    # Skip if target file already exists
    if new_path.exists():
        print(f"  [SKIP] Target exists: {new_path}")
        skipped_files.append(str(file_path))
        return False

    # Rename
    try:
        file_path.rename(new_path)
        renamed_files.append(str(file_path))
        print(f"  [RENAME] {file_path.name} -> {new_name}")
        return True
    except Exception as e:
        print(f"  [ERROR] {file_path.name}: {e}")
        errors.append((str(file_path), str(e)))
        return False

def process_directory(root_dir: Path):
    """Recursively process directory"""
    print(f"\nProcessing directory: {root_dir.relative_to(PROJECT_ROOT)}")

    # Process all files (not directories)
    for item in root_dir.iterdir():
        if item.is_file():
            rename_file(item)
        elif item.is_dir():
            # Recursively process subdirectories
            process_directory(item)

def main():
    print("=" * 60)
    print("Starting cleanup of polluted file names")
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
    print(f"  Renamed files: {len(renamed_files)}")
    print(f"  Skipped files: {len(skipped_files)}")
    print(f"  Error files: {len(errors)}")
    print("=" * 60)

    if errors:
        print("\nError details:")
        for old_path, error in errors:
            print(f"  {old_path}: {error}")

    # Save log
    log_path = PROJECT_ROOT / "rename_cleanup_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Rename Log ===\n\n")
        f.write(f"Renamed files: {len(renamed_files)}\n")
        f.write(f"Skipped files: {len(skipped_files)}\n")
        f.write(f"Error files: {len(errors)}\n\n")

        if renamed_files:
            f.write("--- Renamed files ---\n")
            for path in renamed_files:
                f.write(f"  {path}\n")

        if skipped_files:
            f.write("\n--- Skipped files ---\n")
            for path in skipped_files:
                f.write(f"  {path}\n")

        if errors:
            f.write("\n--- Error files ---\n")
            for old_path, error in errors:
                f.write(f"  {old_path}: {error}\n")

    print(f"\nLog saved to: {log_path}")

if __name__ == "__main__":
    main()
