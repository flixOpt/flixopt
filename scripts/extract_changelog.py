#!/usr/bin/env python3
"""
Extract individual releases from CHANGELOG.md to docs/changelog/
Simple script to create one file per release.
"""

import re
from pathlib import Path

from packaging.version import InvalidVersion, Version
from packaging.version import parse as parse_version


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
    sections = re.split(r'^## \[', content, flags=re.MULTILINE)

    releases = []
    for section in sections[1:]:  # Skip first empty section
        # Extract version and date from start of section
        match = re.match(r'([^\]]+)\] - ([^\n]+)\n(.*)', section, re.DOTALL)
        if match:
            version, date, release_content = match.groups()
            releases.append((version, date.strip(), release_content.strip()))

    print(f'🔍 Found {len(releases)} releases')

    # Sort releases by version (oldest first) to keep existing file prefixes stable.
    def version_key(release):
        try:
            return parse_version(release[0])
        except InvalidVersion:
            return parse_version('0.0.0')  # fallback for invalid versions

    releases.sort(key=version_key, reverse=False)

    # Show what we captured for debugging
    if releases:
        print(f'🔧 First release content length: {len(releases[0][2])}')

    for i, (version_str, date, release_content) in enumerate(releases):
        # Clean up version for filename with numeric prefix (newest first)
        index = 99999 - i  # Newest first, while keeping the same file names for old releases
        prefix = f'{index:05d}'  # Zero-padded 5-digit number
        filename = f'{prefix}-v{version_str.replace(" ", "-")}.md'
        filepath = output_dir / filename

        # Clean up content - remove trailing --- separators and emojis from headers
        cleaned_content = re.sub(r'\s*---\s*$', '', release_content.strip())

        # Generate navigation links
        nav_links = []

        # Previous version (older release)
        if i > 0:
            prev_index = 99999 - (i - 1)
            prev_version = releases[i - 1][0]
            prev_filename = f'{prev_index:05d}-v{prev_version.replace(" ", "-")}.md'
            nav_links.append(f'← [Previous: {prev_version}]({prev_filename})')

        # Next version (newer release)
        if i < len(releases) - 1:
            next_index = 99999 - (i + 1)
            next_version = releases[i + 1][0]
            next_filename = f'{next_index:05d}-v{next_version.replace(" ", "-")}.md'
            nav_links.append(f'[Next: {next_version}]({next_filename}) →')

        # Always add link back to index
        nav_links.append('[📋 All Releases](index.md)')
        # Add GitHub tag link only for valid PEP 440 versions (skip e.g. "Unreleased")
        ver_obj = parse_version(version_str)
        if isinstance(ver_obj, Version):
            nav_links.append(f'[🏷️ GitHub Release](https://github.com/flixOpt/flixopt/releases/tag/v{version_str})')
        # Create content with navigation
        content_lines = [
            f'# {version_str} - {date.strip()}',
            '',
            ' | '.join(nav_links),
            '',
            '---',
            '',
            cleaned_content,
            '',
            '---',
            '',
            ' | '.join(nav_links),
        ]

        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))

        print(f'✅ Created {filename}')

    print(f'🎉 Extracted {len(releases)} releases to docs/changelog/')


def extract_index():
    changelog_path = Path('CHANGELOG.md')
    output_dir = Path('docs/changelog')
    index_path = output_dir / 'index.md'

    if not changelog_path.exists():
        print('❌ CHANGELOG.md not found')
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read changelog
    with open(changelog_path, encoding='utf-8') as f:
        content = f.read()

    intro_match = re.search(r'# Changelog\s+([\s\S]*?)(?=<!--)', content)
    if not intro_match:
        raise ValueError('Intro section not found before comment block')
    final_content = intro_match.group(1).strip()

    # Write simple index without TOC (literate-nav plugin handles navigation)
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(
            '# Changelog\n\n'
            + final_content
            + '\n\nUse the navigation menu to browse through individual release notes.'
        )

    print('✅ Created index.md')


if __name__ == '__main__':
    extract_releases()
    extract_index()
