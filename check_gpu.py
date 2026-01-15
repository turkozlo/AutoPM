import sys

import faiss
import torch


def check_gpu():
    print("--- GPU Availability Check ---")

    # 1. Check Python version
    print(f"Python version: {sys.version}")

    # 2. Check Torch & CUDA
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️ Warning: CUDA not detected. PyTorch will use CPU.")

    # 3. Check FAISS
    print(f"FAISS version: {faiss.__version__}")
    try:
        faiss.StandardGpuResources()
        print("✅ FAISS-GPU is installed and resources are available.")
    except AttributeError:
        print("ℹ️ FAISS-CPU is installed (StandardGpuResources not found).")
    except Exception as e:
        print(f"❌ FAISS-GPU error: {e}")

    print("\n--- Summary ---")
    if cuda_available:
        print("🚀 System is ready for GPU acceleration!")
    else:
        print("💻 System will run on CPU.")


if __name__ == "__main__":
    check_gpu()
