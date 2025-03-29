"""Generate the code reference pages and navigation."""

import sys
from pathlib import Path

import mkdocs_gen_files

# Add the project root to sys.path to ensure modules can be imported
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

nav = mkdocs_gen_files.Nav()

src = root / 'flixopt'
api_dir = 'api-reference'

for path in sorted(src.rglob('*.py')):
    module_path = path.relative_to(src).with_suffix('')
    doc_path = path.relative_to(src).with_suffix('.md')
    full_doc_path = Path(api_dir, doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == '__init__':
        parts = parts[:-1]
        if not parts:
            continue  # Skip the root __init__.py
        doc_path = doc_path.with_name('index.md')
        full_doc_path = full_doc_path.with_name('index.md')
    elif parts[-1] == '__main__' or parts[-1].startswith('_'):
        continue

    # Only add to navigation if there are actual parts
    if parts:
        nav[parts] = doc_path.as_posix()

        # Generate documentation file - always using the flixopt prefix
        with mkdocs_gen_files.open(full_doc_path, 'w') as fd:
            # Use 'flixopt.' prefix for all module references
            module_id = 'flixopt.' + '.'.join(parts)
            fd.write(f'::: {module_id}\n    options:\n       inherited_members: true\n')

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Create an index file for the API reference
with mkdocs_gen_files.open(f'{api_dir}/index.md', 'w') as index_file:
    index_file.write('# API Reference\n\n')
    index_file.write(
        'This section contains the documentation for all modules and classes in flixopt.\n'
        'For more information on how to use the classes and functions, see the [Concepts & Math](../concepts-and-math/index.md) section.\n'
    )

with mkdocs_gen_files.open(f'{api_dir}/SUMMARY.md', 'w') as nav_file:
    nav_file.writelines(nav.build_literate_nav())
