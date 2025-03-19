import os

# Define root directory
ROOT_DIR = r"E:\Programming\OKXsignal"
OUTPUT_FILE = os.path.join(ROOT_DIR, "exported_files.md")

# Allowed file extensions (only source code files are included)
ALLOWED_EXTENSIONS = {".py", ".json", ".md"}

# Any file or directory you want to exclude completely
EXCLUDED_PATHS = {
    os.path.join(ROOT_DIR, "supabase"),
    os.path.join(ROOT_DIR, "config\\credentials.env"),
}

def is_excluded(path):
    """
    Returns True if 'path' is explicitly excluded,
    either as an exact file or because it's within an excluded directory.
    """
    norm_path = os.path.normcase(path)
    for excluded in EXCLUDED_PATHS:
        norm_excluded = os.path.normcase(excluded)

        # Exact match (file or directory)
        if norm_path == norm_excluded:
            return True

        # Path inside an excluded directory
        if norm_path.startswith(norm_excluded + os.sep):
            return True

    return False

def should_include(file_path):
    """Check if a file should be included in the export."""
    if is_excluded(file_path):
        return False

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False

    return True

def export_markdown():
    """Recursively export all relevant code files, respecting excluded paths."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as md_file:
        md_file.write("# Exported Code Files\n\n")

        for root, dirs, files in os.walk(ROOT_DIR):
            # Exclude directories that match EXCLUDED_PATHS
            dirs[:] = [d for d in dirs if not is_excluded(os.path.join(root, d))]

            for filename in files:
                file_path = os.path.join(root, filename)

                if should_include(file_path):
                    relative_path = os.path.relpath(file_path, ROOT_DIR)
                    md_file.write(f"## `{relative_path}`\n\n```python\n")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            md_file.write(f.read())
                    except Exception as e:
                        md_file.write(f"# Skipped {filename}: {e}")
                    
                    md_file.write("\n```\n\n")

    print(f"✅ Exported code to: {OUTPUT_FILE}")

if __name__ == "__main__":
    export_markdown()
