import torch

# 检查 CUDA 是否可用
def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")

        # 获取当前设备的 CUDA 信息
        current_device = torch.cuda.current_device()
        print(f"Current Device ID: {current_device}")
        print(f"Device Name: {torch.cuda.get_device_name(current_device)}")
        print(f"Device Memory: {torch.cuda.get_device_properties(current_device).total_memory / (1024**3):.2f} GB")
        print(f"Multi Processor Count: {torch.cuda.get_device_properties(current_device).multi_processor_count}")
        print(f"CUDA Major Version: {torch.cuda.get_device_properties(current_device).major}")
        print(f"CUDA Minor Version: {torch.cuda.get_device_properties(current_device).minor}")
    else:
        print("CUDA is not available")

if __name__ == "__main__":
    check_cuda()
