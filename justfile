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

# Sweep across git commits (standalone, uses each commit's own benchmark code)
# Examples:
#   just sweep HEAD~10..HEAD --model simple --phase build --sizes 168 --iterations 3
#   just sweep main,feature/perf --model simple district --phase build memory
#   just sweep v1.0,v2.0,HEAD --all --quick
sweep rev_range *args:
    python -m benchmarks.sweep {{rev_range}} {{args}}

# Plot results (unified: rows=phases, cols=models, x=timesteps, color=runs)
plot +files:
    python -m benchmarks.compare {{files}}

# Generate markdown table
table +files:
    python -m benchmarks.compare --table {{files}}

# List available models and phases
list:
    python -m benchmarks.run --list

# Dry-run sweep (show what would be benchmarked)
sweep-dry rev_range *args:
    python -m benchmarks.sweep {{rev_range}} --dry-run {{args}}

# Show recent result files
latest count="10":
    @ls -lt benchmarks/results/*.json 2>/dev/null | head -{{count}} || echo "No results found"

# Clean old results (keeps cache intact)
clean:
    rm -f benchmarks/results/*.json benchmarks/results/*.png benchmarks/results/*.html benchmarks/results/*.md
    @echo "Cleaned benchmarks/results/"

# Clean everything including cache
clean-all: clean
    rm -rf ~/.cache/flixopt-benchmarks/
    @echo "Cleaned ~/.cache/flixopt-benchmarks/"
