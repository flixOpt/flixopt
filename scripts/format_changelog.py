#!/usr/bin/env python3
"""
Format changelog headers to include GitHub links and nicely formatted dates.
Converts from:
    ## [4.0.0] - 2025-11-19
To:
    ## [**v4.0.0**](https://github.com/flixOpt/flixopt/releases/tag/v4.0.0) <small>19th November 2025</small> { id="v4.0.0" }
"""

import re
from datetime import datetime
from pathlib import Path


def format_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to '19th November 2025' format."""
    if '????' in date_str:
        # Keep placeholder dates as-is
        return date_str

    try:
        date = datetime.strptime(date_str.strip(), '%Y-%m-%d')
        day = date.day

        # Add ordinal suffix
        if 4 <= day <= 20 or 24 <= day <= 30:
            suffix = 'th'
        else:
            suffix = ['st', 'nd', 'rd'][day % 10 - 1]

        return f'{day}{suffix} {date.strftime("%B %Y")}'
    except ValueError:
        # If date parsing fails, return original
        return date_str


def format_version_header(match) -> str:
    """Format a version header line."""
    version = match.group(1)
    date_str = match.group(2)

    # Skip Template and Unreleased (keep original format)
    if version in ['Template', 'Unreleased']:
        return match.group(0)

    # Format the date
    formatted_date = format_date(date_str)

    # Create the new header
    github_url = f'https://github.com/flixOpt/flixopt/releases/tag/v{version}'
    new_header = f'## [**{version}**]({github_url}) <small>{formatted_date}</small> {{ id="{version}" }}'

    return new_header


def main():
    """Process the changelog file."""
    changelog_path = Path('docs/changelog.md')

    if not changelog_path.exists():
        print(f'❌ {changelog_path} not found')
        return

    # Read the changelog
    with open(changelog_path, encoding='utf-8') as f:
        content = f.read()

    # Pattern to match version headers: ## [VERSION] - DATE
    pattern = r'^## \[([^\]]+)\] - (.+)$'

    # Replace all version headers
    new_content = re.sub(pattern, format_version_header, content, flags=re.MULTILINE)

    # Write back
    with open(changelog_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f'✅ Formatted changelog headers in {changelog_path}')


if __name__ == '__main__':
    main()
