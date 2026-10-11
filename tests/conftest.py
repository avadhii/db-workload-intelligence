"""
conftest.py — Adds project root to sys.path so that 'whatif' and 'collector'
packages can be imported in tests without installation.
"""
import sys
import os

# Insert project root (one level above /tests/) at the front of sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
