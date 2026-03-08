"""
Delete polluted markdown files
Delete files named: 全网最全51cto课程【缺课漏课、后续更新】+v：_51cto_download.md
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
POLLUTED_SUFFIX = "全网最全51cto课程【缺课漏课、后续更新】+v：_51cto_download.md"

# Statistics
deleted_files = []
errors = []

def delete_file(file_path: Path) -> bool:
    """
    Delete a single polluted file

    Args:
        file_path: File path

    Returns:
        True if deleted successfully
    """
    # Check if it's a polluted file
    if not file_path.name.endswith(POLLUTED_SUFFIX):
        return False

    try:
        file_path.unlink()
        deleted_files.append(str(file_path))
        print(f"  [DELETE] {file_path.name}")
        return True
    except Exception as e:
        print(f"  [ERROR] {file_path.name}: {e}")
        errors.append((str(file_path), str(e)))
        return False

def process_directory(root_dir: Path):
    """Recursively process directory"""
    print(f"\nProcessing directory: {root_dir.relative_to(PROJECT_ROOT)}")

    # Process all files
    for item in root_dir.iterdir():
        if item.is_file() and item.name.endswith(POLLUTED_SUFFIX):
            delete_file(item)
        elif item.is_dir():
            # Recursively process subdirectories
            process_directory(item)

def main():
    print("=" * 60)
    print("Starting deletion of polluted markdown files")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Polluted suffix: {POLLUTED_SUFFIX}")
    print("=" * 60)

    # Recursively process all directories
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir():
            process_directory(item)

    # Print statistics
    print("\n" + "=" * 60)
    print("Deletion completed!")
    print(f"  Deleted files: {len(deleted_files)}")
    print(f"  Error files: {len(errors)}")
    print("=" * 60)

    if errors:
        print("\nError details:")
        for file_path, error in errors:
            print(f"  {file_path}: {error}")

    # Save log
    log_path = PROJECT_ROOT / "delete_polluted_files_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Deletion Log ===\n\n")
        f.write(f"Deleted files: {len(deleted_files)}\n")
        f.write(f"Error files: {len(errors)}\n\n")

        if deleted_files:
            f.write("--- Deleted files ---\n")
            for path in deleted_files:
                f.write(f"  {path}\n")

        if errors:
            f.write("\n--- Error files ---\n")
            for file_path, error in errors:
                f.write(f"  {file_path}: {error}\n")

    print(f"\nLog saved to: {log_path}")

if __name__ == "__main__":
    main()
