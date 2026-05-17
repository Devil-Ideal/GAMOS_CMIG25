import sys
from pathlib import Path


def clean_file(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = [ln.rstrip() for ln in lines if ln.strip()]
    text = "\n".join(new_lines) + ("\n" if new_lines else "")
    if text != path.read_text(encoding="utf-8"):
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main(root: Path) -> None:
    for path in sorted(root.rglob("*.py")):
        if "tools" in path.parts:
            continue
        if clean_file(path):
            print(path.relative_to(root))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
