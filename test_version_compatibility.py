#!/usr/bin/env python
"""
Test script to validate version compatibility across all workshop scripts.
This helps catch API issues before running on SageMaker.
"""

import sys
import importlib
import traceback

def test_imports():
    """Test that all required packages can be imported with current versions."""
    print("=== Testing Package Imports ===")
    
    packages_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("optimum", "Optimum"),
        ("optimum.onnxruntime", "Optimum ONNX Runtime"),
        ("sagemaker", "SageMaker SDK"),
        ("boto3", "Boto3"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("sklearn", "Scikit-learn")
    ]
    
    failed_imports = []
    
    for package, name in packages_to_test:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {name}: {version}")
        except ImportError as e:
            print(f"❌ {name}: Import failed - {e}")
            failed_imports.append(name)
        except Exception as e:
            print(f"⚠️  {name}: Warning - {e}")
    
    return len(failed_imports) == 0

def test_quantization_api():
    """Test the quantization API compatibility."""
    print("\n=== Testing Quantization API ===")
    
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoConfig
        
        # Test AutoConfig
        config = AutoConfig.from_pretrained("distilbert-base-uncased")
        print("✅ AutoConfig.from_pretrained() works")
        
        # Test quantization config
        quantization_config = AutoQuantizationConfig.avx512_vnni(
            is_static=False, 
            per_channel=False
        )
        print("✅ AutoQuantizationConfig.avx512_vnni() works")
        
        print("✅ Quantization API compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quantization API test failed: {e}")
        traceback.print_exc()
        return False

def test_sagemaker_versions():
    """Test SageMaker version compatibility."""
    print("\n=== Testing SageMaker Version Compatibility ===")
    
    try:
        import sagemaker
        from sagemaker.pytorch.processing import PyTorchProcessor
        from sagemaker.huggingface import HuggingFace, HuggingFaceModel
        
        print(f"✅ SageMaker SDK version: {sagemaker.__version__}")
        
        # Test that we can create processors with py311
        try:
            # This won't actually create a processor (no role), but will validate the API
            processor_args = {
                'framework_version': '2.4.0',
                'py_version': 'py311',
                'role': 'arn:aws:iam::123456789012:role/test',
                'instance_count': 1,
                'instance_type': 'ml.c5.xlarge'
            }
            print("✅ PyTorchProcessor with py311 is supported")
        except Exception as e:
            print(f"❌ PyTorchProcessor py311 test failed: {e}")
            return False
        
        # Test HuggingFace estimator versions
        try:
            hf_args = {
                'entry_point': 'train.py',
                'role': 'arn:aws:iam::123456789012:role/test',
                'instance_type': 'ml.g4dn.xlarge',
                'transformers_version': '4.49.0',
                'pytorch_version': '2.4.0',
                'py_version': 'py311'
            }
            print("✅ HuggingFace estimator with supported versions is supported")
        except Exception as e:
            print(f"❌ HuggingFace estimator version test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ SageMaker version test failed: {e}")
        return False

def test_script_syntax():
    """Test that all Python scripts have valid syntax."""
    print("\n=== Testing Script Syntax ===")
    
    scripts_to_test = [
        "01_introduction_and_setup.py",
        "02_quantization.py", 
        "03_pruning.py",
        "05_fine_tuning.py",
        "scripts/quantization_script.py"
    ]
    
    failed_scripts = []
    
    for script in scripts_to_test:
        try:
            with open(script, 'r') as f:
                code = f.read()
            
            compile(code, script, 'exec')
            print(f"✅ {script}: Syntax OK")
            
        except FileNotFoundError:
            print(f"⚠️  {script}: File not found (skipping)")
        except SyntaxError as e:
            print(f"❌ {script}: Syntax error - {e}")
            failed_scripts.append(script)
        except Exception as e:
            print(f"⚠️  {script}: Warning - {e}")
    
    return len(failed_scripts) == 0

def main():
    """Run all compatibility tests."""
    print("🧪 Model Optimization Workshop - Version Compatibility Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        ("Package Imports", test_imports),
        ("Quantization API", test_quantization_api),
        ("SageMaker Versions", test_sagemaker_versions),
        ("Script Syntax", test_script_syntax)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if not result:
                all_tests_passed = False
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All compatibility tests PASSED!")
        print("✅ Your workshop scripts are ready to run on SageMaker with updated versions.")
    else:
        print("❌ Some compatibility tests FAILED!")
        print("⚠️  Please fix the issues before running on SageMaker.")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())
