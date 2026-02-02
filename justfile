default_iterations := "10"

# Run all benchmarks
all name iterations=default_iterations:
    python -m benchmarks.run --name {{name}} --all --iterations {{iterations}}

# Run specific model(s) and/or phase(s)
# Examples:
#   just run main --model simple --phase build
#   just run main --model simple district --phase build memory --sizes 24 168
run name *args:
    python -m benchmarks.run --name {{name}} {{args}}

# Quick smoke test (all models, QUICK_SIZES, 3 iterations)
quick name:
    python -m benchmarks.run --name {{name}} --all --quick --iterations 3

# Compare current branch vs a reference
# Pass extra args to filter models/phases/sizes for both runs
compare ref name="current" iterations=default_iterations *args:
    #!/usr/bin/env bash
    set -euo pipefail
    current_branch=$(git branch --show-current)
    # Benchmark current
    python -m benchmarks.run --name {{name}} --all --iterations {{iterations}} {{args}}
    # Checkout ref, install, benchmark
    git checkout {{ref}}
    pip install -e . --quiet
    python -m benchmarks.run --name {{ref}} --all --iterations {{iterations}} {{args}}
    # Return and compare
    git checkout $current_branch
    pip install -e . --quiet
    python -m benchmarks.compare benchmarks/results/{{name}}_*.json benchmarks/results/{{ref}}_*.json

# Quick compare (fewer iterations)
compare-quick ref *args:
    just compare {{ref}} current 3 {{args}}

# Sweep across git commits (range or comma-separated refs)
# Examples:
#   just sweep HEAD~10..HEAD --model simple --phase build --sizes 168 --iterations 3
#   just sweep HEAD~20..HEAD --model simple --phase build memory --sizes 24 --iterations 3
sweep rev_range *args:
    python -m benchmarks.run --sweep {{rev_range}} {{args}}

# Plot existing results (2-run comparison)
plot +files:
    python -m benchmarks.compare {{files}}

# Plot existing results as sweep timeline
plot-sweep +files:
    python -m benchmarks.compare --sweep {{files}}

# List available models and phases
list:
    python -m benchmarks.run --list
