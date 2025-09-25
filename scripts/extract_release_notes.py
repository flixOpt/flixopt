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

    # Remove template section (HTML comments)
    content = re.sub(r'<!-- This text won\'t be rendered.*?Until here -->', '', content, flags=re.DOTALL)

    # Pattern to match version section: ## **[2.1.9] - 2025-09-23**
    pattern = rf'## \*\*\[{re.escape(version)}\] - [^\*]+\*\*(.*?)(?=^## \*\*\[|^---\s*$|\Z)'
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)

    if not match:
        print(f"❌ Error: No release notes found for version '{version}'", file=sys.stderr)
        sys.exit(1)

    release_content = match.group(1).strip()

    # Clean up content - remove trailing separators
    release_content = re.sub(r'\s*---\s*$', '', release_content)

    return release_content


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
