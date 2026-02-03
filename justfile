default_iterations := "10"

# Run all benchmarks
all name iterations=default_iterations:
    python -m benchmarks.run --name {{name}} --all --iterations {{iterations}}

# Run specific model(s) and/or phase(s)
run name *args:
    python -m benchmarks.run --name {{name}} {{args}}

# Quick smoke test (all models, QUICK_SIZES, 3 iterations)
quick name:
    python -m benchmarks.run --name {{name}} --all --quick --iterations 3

# Sweep across git commits
sweep rev_range *args:
    python -m benchmarks.sweep {{rev_range}} {{args}}

# Dry-run sweep (show what would be benchmarked)
sweep-dry rev_range *args:
    python -m benchmarks.sweep {{rev_range}} --dry-run {{args}}

# Show results as markdown table
table +files:
    python -m benchmarks.compare {{files}}

# Show results as CSV
csv +files:
    python -m benchmarks.compare --csv {{files}}

# List available models and phases
list:
    python -m benchmarks.run --list

# Show recent result files
latest count="10":
    @ls -lt benchmarks/results/*.json 2>/dev/null | head -{{count}} || echo "No results found"

# Clean results
clean:
    rm -f benchmarks/results/*.json benchmarks/results/*.html benchmarks/results/*.md benchmarks/results/*.csv
    @echo "Cleaned benchmarks/results/"

# Clean everything including cache
clean-all: clean
    rm -rf ~/.cache/flixopt-benchmarks/
    @echo "Cleaned ~/.cache/flixopt-benchmarks/"
