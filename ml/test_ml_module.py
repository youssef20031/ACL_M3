"""
Quick Test Script for ML Module
Run before pushing to verify everything works
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from ml.feature_engineering import FeatureEngineer
        from ml.predictor import FPLPredictor
        from ml.api_integration import MLAPIIntegration
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_feature_engineer():
    """Test feature engineer initialization."""
    print("\nTesting FeatureEngineer...")
    try:
        from ml.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        print("✅ FeatureEngineer initialized")
        return True
    except Exception as e:
        print(f"❌ FeatureEngineer failed: {e}")
        return False


def test_predictor_loading():
    """Test predictor loading."""
    print("\nTesting FPLPredictor loading...")
    try:
        from ml.predictor import FPLPredictor
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'linear_regression_v1.pkl')
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}")
            print("   Run 'python ml/train.py' first to train the model")
            return False
        
        predictor = FPLPredictor(model_path=model_path)
        print(f"✅ Predictor loaded from {model_path}")
        print(f"   Model loaded: {predictor.model_loaded}")
        print(f"   Model type: {predictor.model_type}")
        print(f"   Feature names: {len(predictor.feature_engineer.feature_names)} features")
        return True
    except Exception as e:
        print(f"❌ Predictor loading failed: {e}")
        return False


def test_model_files():
    """Test that model files exist."""
    print("\nChecking model files...")
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    required_files = [
        'linear_regression_v1.pkl',
        'linear_regression_v1_mappings.json',
        'training_results.json'
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {filename} ({size} bytes)")
        else:
            print(f"❌ {filename} NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some model files missing. Run 'python ml/train.py' to create them.")
    
    return all_exist


def main():
    """Run all tests."""
    print("=" * 60)
    print("ML Module Test Suite")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "FeatureEngineer": test_feature_engineer(),
        "Model Files": test_model_files(),
        "Predictor Loading": test_predictor_loading()
    }
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready for Git push!")
    else:
        print("❌ SOME TESTS FAILED - Fix issues before pushing")
        print("\nTroubleshooting:")
        print("- If model files missing: Run 'python ml/train.py'")
        print("- If imports fail: Check Python path and dependencies")
        print("- If loading fails: Verify model file integrity")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
