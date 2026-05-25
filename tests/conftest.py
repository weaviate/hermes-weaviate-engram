"""Pytest configuration.

The repo layout puts the plugin under ``weaviate_engram/`` (a regular
package) and tests under ``tests/``. ``pythonpath = "."`` in pyproject
makes ``import weaviate_engram`` resolve naturally — nothing to do here
yet, but this file is the natural home for any future shared fixtures.
"""
