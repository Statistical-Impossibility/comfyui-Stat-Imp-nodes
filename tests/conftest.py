import os
import sys

# Make the pack modules (color_core, color_nodes, nodes, ...) importable when
# running `pytest` from the pack dir, without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
