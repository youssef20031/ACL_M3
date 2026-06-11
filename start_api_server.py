"""
Quick start script for API server with dependency workaround
Starts uvicorn server on port 8000
"""
import os
import sys

# Set environment variables BEFORE any imports
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Try to install tf-keras if missing
try:
    import tf_keras
    print("✅ tf-keras is installed")
except ImportError:
    print("⚠️  tf-keras not found - attempting to install...")
    os.system("pip install tf-keras")
    print("✅ tf-keras installed")

print("\nStarting API server on port 8000...")
print("Press CTRL+C to stop the server\n")

# Start uvicorn
os.system("uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload")
