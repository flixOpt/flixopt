#!/usr/bin/env python3
"""
Extract release notes from CHANGELOG.md for a specific version.
Usage: python extract_release_notes.py <version>
"""

import re
import sys
from pathlib import Path


def extract_release_notes(version: str) -> str:
    """Extract release notes for a specific version from CHANGELOG.md"""
    changelog_path = Path('CHANGELOG.md')

    if not changelog_path.exists():
        print('❌ Error: CHANGELOG.md not found', file=sys.stderr)
        sys.exit(1)

    content = changelog_path.read_text(encoding='utf-8')

    # Pattern to match version section: ## [2.1.2] - 2025-06-14
    pattern = rf'## \[{re.escape(version)}\] - [^\n]+\n(.*?)(?=\n## \[|\n\[Unreleased\]|\Z)'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print(f"❌ Error: No release notes found for version '{version}'", file=sys.stderr)
        sys.exit(1)

    return match.group(1).strip()


def main():
    if len(sys.argv) != 2:
        print('Usage: python extract_release_notes.py <version>')
        print('Example: python extract_release_notes.py 2.1.2')
        sys.exit(1)

    version = sys.argv[1]
    release_notes = extract_release_notes(version)
    print(release_notes)


if __name__ == '__main__':
    main()
