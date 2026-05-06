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
    files = []
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
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "extension": file_path.suffix or "[no ext]",
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("modified", ascending=False)


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name in TEXT_EXTENSIONS


def inject_auto_refresh(seconds: int):
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


def format_datetime(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def main():
    st.set_page_config(page_title="Work Dashboard", layout="wide")

    st.title("Work Dashboard")
    st.caption("Live workspace view with file tracking and previews")

    with st.sidebar:
        st.header("Controls")

        auto_refresh = st.checkbox("Auto refresh", value=True)
        refresh_seconds = st.slider("Interval (seconds)", 5, 60, 10)

        if st.button("Refresh now"):
            st.rerun()

        show_hidden = st.checkbox("Show hidden files", value=False)

        excluded_dirs = st.multiselect(
            "Exclude directories",
            options=sorted(DEFAULT_EXCLUDED_DIRS),
            default=sorted(DEFAULT_EXCLUDED_DIRS),
        )

        search_query = st.text_input("Search file")

    if auto_refresh:
        inject_auto_refresh(refresh_seconds)

    files = iter_project_files(ROOT, excluded_dirs)

    if not show_hidden:
        files = [f for f in files if not any(part.startswith(".") for part in f.parts)]

    if search_query:
        files = [f for f in files if search_query.lower() in str(f).lower()]

    st.metric("Files detected", len(files))

    if not files:
        st.warning("No files found")
        return

    df = build_file_table(files)
    df["modified"] = df["modified"].apply(format_datetime)

    st.subheader("Project Files")
    st.dataframe(df, use_container_width=True, hide_index=True)

    selected_path = st.selectbox("Select file to preview", df["path"].tolist())
    selected_file = ROOT / selected_path

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Preview")
        if is_text_file(selected_file):
            content = selected_file.read_text(encoding="utf-8", errors="replace")
            st.code(content[:15000], language=selected_file.suffix.lstrip(".") or "text")
        else:
            st.info("Preview available only for text files")

    with col2:
        st.subheader("Details")
        stat = selected_file.stat()
        st.write(f"Path: {selected_path}")
        st.write(f"Size: {round(stat.st_size / 1024, 2)} KB")
        st.write(f"Modified: {format_datetime(datetime.fromtimestamp(stat.st_mtime))}")
        st.write(f"Extension: {selected_file.suffix or '[no ext]'}")


if __name__ == "__main__":
    main()
