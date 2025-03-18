import os

# Define root directory
ROOT_DIR = "E:\\Programming\\OKXsignal"
OUTPUT_FILE = os.path.join(ROOT_DIR, "exported_files.md")

# Allowed file extensions (Ensuring only source code files are included)
ALLOWED_EXTENSIONS = {".py", ".json", ".md"}

# ✅ Directories that MUST be included
INCLUDED_DIRS = {
    "backend",
    "backend\\api",
    "backend\\config",
    "backend\\execution",
    "backend\\indicators",
    "backend\\signal_engine",
    "backend\\utils",
    "dashboard",
    "okx_api",
    "main.py",
}

# ✅ Explicitly Exclude frontend/grafana
EXCLUDED_DIRS = {
    "frontend\\grafana",
}

def should_include(file_path):
    """Check if a file should be included in the export."""
    if os.path.splitext(file_path)[1] not in ALLOWED_EXTENSIONS:
        return False
    return True

def export_markdown():
    """Exports only relevant code files, ensuring frontend is included properly."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as md_file:
        md_file.write("# Exported Code Files\n\n")  # ✅ Ensures only relevant files are included

        for included_dir in INCLUDED_DIRS:
            full_dir_path = os.path.join(ROOT_DIR, included_dir)

            if not os.path.exists(full_dir_path):
                print(f"⚠️ Skipping missing directory: {included_dir}")
                continue

            # ✅ Skip explicitly excluded directories
            if any(excluded in full_dir_path for excluded in EXCLUDED_DIRS):
                continue

            # ✅ Print folder name before listing its files
            md_file.write(f"## 📂 {included_dir}\n\n")

            try:
                files = [f for f in os.listdir(full_dir_path) if should_include(f)]
            except Exception as e:
                md_file.write(f"⚠️ Skipped {included_dir} due to error: {e}\n\n")
                continue

            for file in files:
                file_path = os.path.join(full_dir_path, file)

                md_file.write(f"### `{file}`\n\n```python\n")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        md_file.write(f.read())
                except Exception as e:
                    md_file.write(f"# Skipped {file}: {e}")
                
                md_file.write("\n```\n\n")

    print(f"✅ Exported code to: {OUTPUT_FILE}")

if __name__ == "__main__":
    export_markdown()
