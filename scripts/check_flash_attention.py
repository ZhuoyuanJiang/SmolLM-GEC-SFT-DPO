#!/usr/bin/env python3
"""
Flash Attention 2 Diagnostic Tool
Checks if Flash Attention 2 is properly installed and available for use
"""

import torch
import sys


def check_gpu():
    """Check GPU availability and configuration"""
    print("=== GPU DIAGNOSTIC ===")
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        print(f"GPU Name: {name} | Compute Capability: {cc[0]}.{cc[1]}")
    else:
        print("No CUDA GPU available")
        return False
    
    # Test tensor on GPU
    try:
        test_tensor = torch.tensor([1.0, 2.0]).cuda()
        print(f"✅ GPU tensor test successful: {test_tensor.device}")
    except Exception as e:
        print(f"❌ GPU tensor test failed: {e}")
        return False
    
    return True


def check_flash_attention():
    """Check Flash Attention 2 availability"""
    print("\n=== FlashAttention 2 Diagnostics ===")
    fa_ok = False
    
    try:
        import flash_attn
        ver = getattr(flash_attn, "__version__", "unknown")
        print(f"flash-attn version: {ver}")
        
        try:
            major = int(str(ver).split(".")[0])
        except Exception:
            major = None
        
        if major and major >= 2:
            print("✅ FlashAttention 2 detected")
            fa_ok = True
        elif major:
            print("⚠️ FlashAttention < 2 detected; upgrade recommended")
        else:
            print("⚠️ Unable to parse flash-attn version; upgrade recommended")
        
        # Quick kernel import test (sanity check)
        try:
            from flash_attn import flash_attn_qkvpacked_func
            print("✅ Kernels import test passed")
        except Exception as e:
            print(f"⚠️ Kernel import test failed: {e}")
    
    except Exception as e:
        print(f"❌ flash-attn not importable: {e}")
    
    return fa_ok


def check_transformers_integration():
    """Check if FlashAttention 2 works with transformers"""
    print("\n=== Transformers Integration Test ===")
    
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
        
        # Try loading a model with FlashAttention 2
        from transformers import AutoModelForCausalLM
        
        print("Testing SmolLM-135M with FlashAttention 2...")
        try:
            m = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceTB/SmolLM-135M",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
                device_map="auto",
            )
            
            attn_impl = getattr(m.config, "attn_implementation", 
                               getattr(m.config, "_attn_implementation", "unknown"))
            print(f"✅ Loaded with FA2: {attn_impl}")
            
            # Cleanup
            import gc
            del m
            gc.collect()
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"⚠️ FA2 load failed: {e}")
            print("Model will fall back to standard attention")
            return False
    
    except Exception as e:
        print(f"Transformers import error: {e}")
        return False


def print_installation_guide():
    """Print installation instructions if FA2 is not available"""
    print("\n=== Installation Guide ===")
    print("To install FlashAttention 2:")
    print("1) pip install --upgrade pip wheel setuptools")
    print("2) pip install -U flash-attn --no-build-isolation")
    print(f"\nEnsure CUDA matches PyTorch (CUDA {torch.version.cuda}) and try:")
    print("   export CUDA_HOME=/usr/local/cuda-12.1  # adjust if needed")
    print("\nFor batch-level packing experiments:")
    print("- FlashAttention 2 is HIGHLY RECOMMENDED for efficiency")
    print("- Without FA2, batch-level packing will use DataCollatorWithFlattening (slower)")


def main():
    """Main diagnostic function"""
    print("=" * 60)
    print("FlashAttention 2 Diagnostic Tool")
    print("=" * 60)
    
    # Check GPU
    gpu_ok = check_gpu()
    if not gpu_ok:
        print("\n❌ GPU not available. FlashAttention requires CUDA.")
        sys.exit(1)
    
    # Check Flash Attention
    fa_ok = check_flash_attention()
    
    # Check Transformers integration
    integration_ok = check_transformers_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if fa_ok and integration_ok:
        print("✅ FlashAttention 2 is fully functional!")
        print("✅ Batch-level packing with padding_free=True will be available")
    elif fa_ok and not integration_ok:
        print("⚠️ FlashAttention 2 installed but transformers integration failed")
        print("⚠️ Will use DataCollatorWithFlattening for batch-level packing")
    else:
        print("❌ FlashAttention 2 not available")
        print("⚠️ Will use DataCollatorWithFlattening for batch-level packing (slower)")
        print_installation_guide()
    
    return fa_ok and integration_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)