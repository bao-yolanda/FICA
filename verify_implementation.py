"""
Simple verification script for quantile policy implementation
This checks syntax and basic structure without running full optimization
"""

import ast
import os

def verify_file_syntax(filepath):
    """Verify Python file syntax"""
    print(f"\nVerifying: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"  ✓ Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False

def check_file_exists(filepath):
    """Check if file exists"""
    if os.path.exists(filepath):
        print(f"  ✓ File exists")
        return True
    else:
        print(f"  ✗ File not found")
        return False

def check_imports(filepath):
    """Check if required imports are present"""
    required_imports = ['numpy', 'matplotlib', 'joblib']
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    missing = []
    for imp in required_imports:
        if imp not in content:
            missing.append(imp)

    if missing:
        print(f"  ⚠ Missing imports: {missing}")
        return False
    else:
        print(f"  ✓ All required imports present")
        return True

def check_class_structure(filepath, class_name):
    """Check if class exists"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if f"class {class_name}" in content:
        print(f"  ✓ Class {class_name} found")
        return True
    else:
        print(f"  ✗ Class {class_name} not found")
        return False

def main():
    print("=" * 70)
    print("QUANTILE POLICY - IMPLEMENTATION VERIFICATION")
    print("=" * 70)

    files_to_check = [
        'quantile_policy.py',
        'evaluate_quantile.py',
        'test_quantile.py',
    ]

    all_valid = True

    for filename in files_to_check:
        filepath = os.path.join(os.getcwd(), filename)
        print(f"\n{'='*70}")
        print(f"Checking: {filename}")
        print('='*70)

        if not check_file_exists(filepath):
            all_valid = False
            continue

        if not verify_file_syntax(filepath):
            all_valid = False
            continue

        if not check_imports(filepath):
            all_valid = False
            continue

        # Check specific classes
        if filename == 'quantile_policy.py':
            check_class_structure(filepath, 'QuantilePolicy')
        elif filename == 'evaluate_quantile.py':
            check_class_structure(filepath, 'QuantileEvaluator')

    print("\n" + "=" * 70)
    if all_valid:
        print("✓ ALL CHECKS PASSED!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Ensure FICA environment is activated")
        print("2. Run: python test_quantile.py")
        print("3. Or for full evaluation: python evaluate_quantile.py")
    else:
        print("✗ SOME CHECKS FAILED!")
        print("=" * 70)
    print("")

if __name__ == '__main__':
    main()
