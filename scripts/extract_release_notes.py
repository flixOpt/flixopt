#!/usr/bin/env python3
"""
Extract release notes from CHANGELOG.md for a specific version.
Usage: python extract_release_notes.py <version>
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


def extract_release_notes(version: str, changelog_path: Path = Path("CHANGELOG.md")) -> str:
    """
    Extract release notes for a specific version from CHANGELOG.md

    Args:
        version: Version string (e.g., "2.1.2")
        changelog_path: Path to the changelog file

    Returns:
        Release notes content as string

    Raises:
        FileNotFoundError: If changelog file doesn't exist
        ValueError: If version section not found
    """
    if not changelog_path.exists():
        raise FileNotFoundError(f"Changelog file not found: {changelog_path}")

    content = changelog_path.read_text(encoding='utf-8')

    # Pattern to match version section in Keep a Changelog format
    # Matches: ## [2.1.2] - 2025-06-14
    pattern = rf'## \[{re.escape(version)}\] - [^\n]+\n(.*?)(?=\n## \[|\n\[Unreleased\]|\Z)'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        # Try alternative format without brackets: ## 2.1.2 - 2025-06-14
        pattern_alt = rf'## {re.escape(version)} - [^\n]+\n(.*?)(?=\n## |\Z)'
        match = re.search(pattern_alt, content, re.DOTALL)

    if not match:
        available_versions = extract_available_versions(content)
        raise ValueError(
            f"No release notes found for version '{version}'. "
            f"Available versions: {', '.join(available_versions)}"
        )

    release_notes = match.group(1).strip()

    # Clean up the content
    release_notes = clean_release_notes(release_notes)

    return release_notes

def extract_available_versions(content: str) -> list[str]:
    """Extract all available version numbers from changelog."""
    # Match both [version] and version formats
    patterns = [
        r'## \[([^\]]+)\] - ',  # [2.1.2] format
        r'## ([0-9]+\.[0-9]+\.[0-9]+) - '  # 2.1.2 format
    ]

    versions = []
    for pattern in patterns:
        versions.extend(re.findall(pattern, content))

    # Remove "Unreleased" if present and deduplicate
    versions = [v for v in set(versions) if v.lower() != 'unreleased']

    # Sort versions (basic string sort, good enough for display)
    return sorted(versions, reverse=True)

def clean_release_notes(content: str) -> str:
    """Clean up release notes content for GitHub release."""
    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

    # Ensure proper spacing after headers
    content = re.sub(r'(### [^\n]+)\n([^\n])', r'\1\n\n\2', content)

    return content.strip()

def main():
    parser = argparse.ArgumentParser(
        description="Extract release notes from CHANGELOG.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_release_notes.py 2.1.2
  python extract_release_notes.py 2.1.2 --changelog docs/CHANGELOG.md
  python extract_release_notes.py 2.1.2 --output release_notes.md
        """
    )

    parser.add_argument(
        'version',
        help='Version to extract (e.g., "2.1.2")'
    )

    parser.add_argument(
        '--changelog',
        type=Path,
        default=Path('CHANGELOG.md'),
        help='Path to changelog file (default: CHANGELOG.md)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output file path (default: stdout)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate that the version exists, don\'t output content'
    )

    args = parser.parse_args()

    try:
        release_notes = extract_release_notes(args.version, args.changelog)

        if args.validate_only:
            print(f"✅ Release notes found for version {args.version}")
            return 0

        if args.output:
            args.output.write_text(release_notes, encoding='utf-8')
            print(f"✅ Release notes written to {args.output}")
        else:
            print(release_notes)

        return 0

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
