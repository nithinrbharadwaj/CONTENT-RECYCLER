"""
conftest.py
===========
Pytest configuration — ensures the project root is on sys.path so
`from src.xxx import yyy` works correctly during test runs.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
