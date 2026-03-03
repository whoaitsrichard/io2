#!/bin/bash
set -e

cd "$(dirname "$0")"
mkdir -p outputs

echo "=========================================="
echo "Running Exercise 1 (ps2_ex1.py)"
echo "=========================================="
python ps2_ex1.py

echo ""
echo "=========================================="
echo "Running Exercise 2 (ps2_ex2.py)"
echo "=========================================="
python ps2_ex2.py

echo ""
echo "=========================================="
echo "Running Exercise 3 (ps2_ex3.py)"
echo "=========================================="
python ps2_ex3.py

echo ""
echo "=========================================="
echo "All exercises complete."
echo "=========================================="
