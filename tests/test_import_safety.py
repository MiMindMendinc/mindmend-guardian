"""Verify runtime modules remain import-safe without hardware dependencies."""

import importlib


def test_mindmend_guardian_import_is_side_effect_free():
    module = importlib.import_module("guardian.mindmend_guardian")
    assert hasattr(module, "main")
    assert hasattr(module, "run_guardian_loop")


def test_luna_app_factory_imports_without_server_start():
    module = importlib.import_module("guardian.luna.app")
    assert hasattr(module, "create_app")


def test_demo_module_imports():
    module = importlib.import_module("guardian.demo.cli")
    assert hasattr(module, "run_demo")
