# Test PyTorch GPU support
print("\n🎮 Testing PyTorch GPU support...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"   ✅ CUDA device count: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"   Device {i}: {device_name}")
        
        # Test a simple GPU operation
        try:
            device = torch.device('cuda:0')
            x = torch.ones(100, 100, device=device)
            result = torch.sum(x)
            print(f"   ✅ GPU computation test passed: sum = {result}")
        except Exception as e:
            print(f"   ❌ GPU computation test failed: {e}")
    else:
        print("   ⚠️  CUDA not available - will run on CPU")
        
except ImportError as e:
    print(f"   ❌ PyTorch not installed: {e}")
except Exception as e:
    print(f"   ❌ PyTorch test error: {e}")
