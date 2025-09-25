#!/usr/bin/env python3
"""
Extract individual releases from CHANGELOG.md to docs/changelog/
Simple script to create one file per release.
"""

import os
import re
from pathlib import Path


def extract_releases():
    """Extract releases from CHANGELOG.md and save to individual files."""

    changelog_path = Path('CHANGELOG.md')
    output_dir = Path('docs/changelog')

    if not changelog_path.exists():
        print('❌ CHANGELOG.md not found')
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read changelog
    with open(changelog_path, encoding='utf-8') as f:
        content = f.read()

    # Remove template section (HTML comments)
    content = re.sub(r'<!-- This text won\'t be rendered.*?Until here -->', '', content, flags=re.DOTALL)

    # Split by release headers
    sections = re.split(r'^## \*\*\[', content, flags=re.MULTILINE)

    releases = []
    for section in sections[1:]:  # Skip first empty section
        # Extract version and date from start of section
        match = re.match(r'([^\]]+)\] - ([^\*]+)\*\*(.*)', section, re.DOTALL)
        if match:
            version, date, release_content = match.groups()
            releases.append((version, date.strip(), release_content.strip()))

    print(f'🔍 Found {len(releases)} releases')

    # Show what we captured for debugging
    if releases:
        print(f'🔧 First release content length: {len(releases[0][2])}')

    for version, date, release_content in releases:
        # Clean up version for filename
        filename = f'v{version.replace(" ", "-")}.md'
        filepath = output_dir / filename

        # Clean up content - remove trailing --- separators and emojis from headers
        cleaned_content = re.sub(r'\s*---\s*$', '', release_content.strip())
        # Remove emojis from section headers like "### ✨ Added" -> "### Added"
        cleaned_content = re.sub(r'^(###\s+)[^\s]+\s+', r'\1', cleaned_content, flags=re.MULTILINE)

        # Create content
        content_lines = [f'# {version} - {date.strip()}', '', cleaned_content]

        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))

        print(f'✅ Created {filename}')

    print(f'🎉 Extracted {len(releases)} releases to docs/changelog/')


if __name__ == '__main__':
    extract_releases()
