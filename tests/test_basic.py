"""
Basic test suite for MindMend Guardian

Run with: python -m pytest tests/
or: python tests/test_basic.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_imports():
    """Test that all main modules can be compiled without syntax errors"""
    import py_compile
    
    files_to_test = [
        'guardian/mindmend_guardian.py',
        'guardian/luna/luna_safety_core.py',
        'perrien-simulator/app.py',
        'tools/test_tts.py',
    ]
    
    for filepath in files_to_test:
        try:
            py_compile.compile(filepath, doraise=True)
            print(f"✓ {filepath}: Syntax OK")
        except py_compile.PyCompileError as e:
            print(f"✗ {filepath}: Syntax Error")
            raise


def test_package_structure():
    """Test that package __init__ files exist"""
    init_files = [
        'guardian/__init__.py',
        'guardian/luna/__init__.py',
        'perrien-simulator/__init__.py',
    ]
    
    for init_file in init_files:
        assert os.path.exists(init_file), f"Missing {init_file}"
        print(f"✓ {init_file}: Exists")


def test_requirements_file():
    """Test that requirements.txt exists and points to project metadata."""
    assert os.path.exists('requirements.txt'), "requirements.txt is missing"

    with open('requirements.txt', 'r') as f:
        content = f.read()
        assert len(content) > 0, "requirements.txt is empty"
        assert 'pyproject.toml' in content or '-e .' in content, "requirements.txt should reference editable install"

    assert os.path.exists('pyproject.toml'), "pyproject.toml is missing"
    print("✓ requirements.txt: Valid")


def test_readme_exists():
    """Test that README.md exists"""
    assert os.path.exists('README.md'), "README.md is missing"
    print("✓ README.md: Exists")


def test_models_directory():
    """Test that models directory exists"""
    assert os.path.exists('models'), "models directory is missing"
    assert os.path.isdir('models'), "models should be a directory"
    print("✓ models/: Directory exists")


if __name__ == '__main__':
    """Run tests directly without pytest"""
    print("Running MindMend Guardian Basic Tests\n")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Package Structure", test_package_structure),
        ("Requirements File", test_requirements_file),
        ("README", test_readme_exists),
        ("Models Directory", test_models_directory),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 60)
        try:
            test_func()
            passed += 1
            print(f"✓ {name} PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {name} FAILED: {e}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)
