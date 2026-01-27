#!/bin/bash

# Run all problem set files in order

set -e  # Exit on any error

echo "Running q2.py..."
python q2.py

echo "Running q3_5.py..."
python q3_5.py

echo "Running q4.py..."
python q4.py

echo "Running pyblp_q4.py..."
python pyblp_q4.py

echo "Plotting joint distributions..."
python "Plot Joint Distributions from PyBLP an.py"

echo "All scripts completed successfully."
