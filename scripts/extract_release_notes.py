#!/usr/bin/env python3
"""
Extract release notes from CHANGELOG.md for a specific version.
Usage: python extract_release_notes.py <version>
"""

import re
import sys
from pathlib import Path


def extract_release_notes(version: str) -> str:
    """Extract release notes for a specific version from CHANGELOG.md.

    For pre-release versions (rc, alpha, beta), falls back to base version notes.
    E.g., 6.0.0rc5 will use notes from 6.0.0 if no specific rc5 entry exists.
    """
    changelog_path = Path('CHANGELOG.md')

    if not changelog_path.exists():
        print('❌ Error: CHANGELOG.md not found', file=sys.stderr)
        sys.exit(1)

    content = changelog_path.read_text(encoding='utf-8')

    # Try exact version first, then fall back to base version for pre-releases
    versions_to_try = [version]
    base_version = re.sub(r'(rc|alpha|beta)\d*$', '', version)
    if base_version != version:
        versions_to_try.append(base_version)

    for v in versions_to_try:
        # Pattern to match version section: ## [2.1.2] - 2025-06-14 or ## [6.0.0] - Upcoming
        pattern = rf'## \[{re.escape(v)}\] - [^\n]+\n(.*?)(?=\n## \[|\n\[Unreleased\]|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

    print(f"❌ Error: No release notes found for version '{version}'", file=sys.stderr)
    sys.exit(1)


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
