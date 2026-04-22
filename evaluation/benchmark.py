"""
evaluation/benchmark.py
=======================
NOTE: This file is retained for import compatibility.
The canonical benchmark runner is `run_benchmark.py` in the project root.

Usage:
    python run_benchmark.py 200

For visualization of benchmark results:
    python -m evaluation.visualize
"""

# Re-export the main benchmark function so existing imports still work
import sys
import os

# Make project root importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Note: run_benchmark is in root-level run_benchmark.py
# To run benchmark, use: python run_benchmark.py <num_samples>

if __name__ == "__main__":
    print("⚠️ Use the root-level runner instead:")
    print("   python run_benchmark.py 200")
