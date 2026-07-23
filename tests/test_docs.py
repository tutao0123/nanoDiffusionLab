import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
EXERCISE_HEADING = re.compile(r"^#{1,6}\s+.*练习", re.MULTILINE)


def markdown_files() -> list[Path]:
    excluded = {".git", ".pytest_cache", ".ruff_cache", "out"}
    return [
        path
        for path in ROOT.rglob("*.md")
        if not any(part in excluded for part in path.relative_to(ROOT).parts)
    ]


def test_local_markdown_links_exist() -> None:
    missing: list[str] = []
    for document in markdown_files():
        text = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            local_path = target.split("#", maxsplit=1)[0]
            if local_path and not (document.parent / local_path).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "Broken local Markdown links:\n" + "\n".join(missing)


def test_readme_languages_are_separate() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    assert not CJK.search(english)
    assert CJK.search(chinese)


def test_tutorial_has_no_exercise_sections() -> None:
    for tutorial in sorted((ROOT / "tutorials" / "zh-CN").glob("*.md")):
        text = tutorial.read_text(encoding="utf-8")
        assert not EXERCISE_HEADING.search(text), tutorial.relative_to(ROOT)
