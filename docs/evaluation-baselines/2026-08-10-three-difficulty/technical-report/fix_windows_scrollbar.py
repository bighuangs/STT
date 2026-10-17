from __future__ import annotations

import sys
from pathlib import Path


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
    if len(sys.argv) != 2:
        raise SystemExit("usage: fix_windows_scrollbar.py <portable-report.html>")
    path = Path(sys.argv[1]).resolve()
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
