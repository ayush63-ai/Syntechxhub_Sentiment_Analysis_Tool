"""Tests for internal module imports and documented project layout."""

import ast
from pathlib import Path


def test_training_pipeline_imports_root_preprocessing_module():
    train_path = Path(__file__).with_name("train.py")
    syntax_tree = ast.parse(train_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "preprocessing" in imported_modules
    assert "utils.preprocessing" not in imported_modules
