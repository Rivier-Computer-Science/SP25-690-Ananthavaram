from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "saved_emotion_model",
    "results_emotion_model",
}
TEXT_EXTENSIONS = {
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".gitignore",
}


def iter_project_files(root: Path, excluded_dirs: Iterable[str]) -> list[Path]:
    excluded = set(excluded_dirs)
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in excluded for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: str(item.relative_to(root)).lower())


def build_file_table(files: list[Path]) -> pd.DataFrame:
    rows = []
    for file_path in files:
        stat = file_path.stat()
        rows.append(
            {
                "path": str(file_path.relative_to(ROOT)),
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "extension": file_path.suffix or "[no ext]",
            }
        )
    return pd.DataFrame(rows)


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name in TEXT_EXTENSIONS


def inject_auto_refresh(seconds: int) -> None:
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {seconds * 1000});
        </script>
        """,
        height=0,
    )


def main() -> None:
    st.set_page_config(page_title="Work Dashboard", page_icon=":open_file_folder:", layout="wide")
    st.title("Work Dashboard")
    st.caption("Live view of the project workspace with refresh controls and file previews.")

    with st.sidebar:
        st.header("Refresh")
        auto_refresh = st.checkbox("Auto refresh", value=True)
        refresh_seconds = st.slider("Interval (seconds)", min_value=5, max_value=60, value=10)
        if st.button("Refresh now", use_container_width=True):
            st.rerun()

        st.header("File View")
        show_hidden = st.checkbox("Show hidden files", value=False)
        excluded_dirs = st.multiselect(
            "Skip directories",
            options=sorted(DEFAULT_EXCLUDED_DIRS),
            default=sorted(DEFAULT_EXCLUDED_DIRS),
        )

    if auto_refresh:
        inject_auto_refresh(refresh_seconds)

    files = iter_project_files(ROOT, excluded_dirs)
    if not show_hidden:
        files = [file_path for file_path in files if not any(part.startswith(".") for part in file_path.parts)]

    st.metric("Files in view", len(files))

    if not files:
        st.warning("No files matched the current filters.")
        return

    file_table = build_file_table(files)
    st.subheader("Workspace Files")
    st.dataframe(file_table, use_container_width=True, hide_index=True)

    selected_relative_path = st.selectbox("Preview a file", options=file_table["path"].tolist())
    selected_file = ROOT / selected_relative_path

    preview_col, details_col = st.columns([2, 1])
    with preview_col:
        st.subheader(f"Preview: {selected_relative_path}")
        if is_text_file(selected_file):
            preview_text = selected_file.read_text(encoding="utf-8", errors="replace")
            st.code(preview_text[:12000], language=selected_file.suffix.lstrip(".") or "text")
        else:
            st.info("Preview is available for common text files. This file is shown as metadata only.")

    with details_col:
        st.subheader("File Details")
        stat = selected_file.stat()
        st.write(f"Path: `{selected_relative_path}`")
        st.write(f"Size: `{round(stat.st_size / 1024, 2)} KB`")
        st.write(
            "Modified: "
            f"`{datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        st.write(f"Extension: `{selected_file.suffix or '[no ext]'}`")


if __name__ == "__main__":
    main()
