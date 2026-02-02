default_iterations := "10"

# Run all benchmarks
all name iterations=default_iterations:
    python -m benchmarks.run --name {{name}} --all --iterations {{iterations}}

# Run specific model+phase
model name model phase iterations=default_iterations:
    python -m benchmarks.run --name {{name}} --model {{model}} --phase {{phase}} --iterations {{iterations}}

# Quick smoke test
quick name:
    python -m benchmarks.run --name {{name}} --all --quick --iterations 3

# Compare current branch vs a reference
compare ref name="current" iterations=default_iterations:
    #!/usr/bin/env bash
    set -euo pipefail
    current_branch=$(git branch --show-current)
    # Benchmark current
    python -m benchmarks.run --name {{name}} --all --iterations {{iterations}}
    # Checkout ref, install, benchmark
    git checkout {{ref}}
    pip install -e . --quiet
    python -m benchmarks.run --name {{ref}} --all --iterations {{iterations}}
    # Return and compare
    git checkout $current_branch
    pip install -e . --quiet
    python -m benchmarks.compare benchmarks/results/{{name}}_*.json benchmarks/results/{{ref}}_*.json

# Quick compare (fewer iterations)
compare-quick ref:
    just compare {{ref}} current 3

# Plot existing results
plot +files:
    python -m benchmarks.compare {{files}}

# List available models and phases
list:
    python -m benchmarks.run --list
