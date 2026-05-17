import io
import sys
import tokenize
from pathlib import Path


def strip_comments(source: str) -> str:
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            continue
        out.append(tok)
    return tokenize.untokenize(out)


def main(root: Path) -> None:
    for path in sorted(root.rglob("*.py")):
        if "tools" in path.parts and path.name == "strip_comments.py":
            continue
        text = path.read_text(encoding="utf-8")
        new_text = strip_comments(text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            print(f"updated: {path.relative_to(root)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
