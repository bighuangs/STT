from __future__ import annotations

import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent
STYLE = """<style data-windows-scrollbar-fix>
html, body {
  max-width: 100% !important;
  overflow-x: clip !important;
}
@media screen and (min-width:761px) {
  .portable-page-header {
    width: 100% !important;
    margin-right: 0 !important;
    margin-left: 0 !important;
  }
}
</style>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the bounded Windows 100vw scrollbar compatibility style to a built portable report."
    )
    parser.add_argument("report", help="HTML file inside this technical-report directory")
    args = parser.parse_args()
    candidate = Path(args.report)
    path = candidate.resolve() if candidate.is_absolute() else (HERE / candidate).resolve()
    try:
        path.relative_to(HERE)
    except ValueError as exc:
        raise SystemExit("report must stay inside remediation-2026-08-10/technical-report") from exc
    if not path.is_file():
        raise SystemExit(f"report does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if "data-windows-scrollbar-fix" in text:
        print(f"Already patched {path}")
        return
    marker = "</head>"
    if marker not in text:
        raise SystemExit("portable report has no </head> marker")
    path.write_text(text.replace(marker, STYLE + marker, 1), encoding="utf-8")
    print(f"Patched Windows 100vw scrollbar overflow in {path}")


if __name__ == "__main__":
    main()
