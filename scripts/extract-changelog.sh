#!/bin/bash
set -e

# Extract changelog releases to docs/changelog/
echo "📝 Extracting changelog releases..."

python3 scripts/extract-changelog.py

echo "✅ Done! Files created in docs/changelog/"
