import warnings

warnings.filterwarnings('ignore')
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载模型
    model = YOLO('ultralytics/cfg/models/ours/yolo11-HFA+DEConv.yaml')

    # 2. 打印基础信息 (可选)
    # model.info(detailed=True)

    # ================= 核心修改部分：打印每层输出通道数 =================
    print("\n" + "=" * 90)
    print(f"{'Layer':<5} | {'Module Name':<25} | {'Out Channels':<15} | {'Output Shape (B, C, H, W)'}")
    print("-" * 90)

    # 定义钩子函数：用于在网络前向传播时截获并打印输出形状
    hooks = []


    def register_hook(i):
        def hook(module, input, output):
            # 处理输出可能是元组的情况 (如 Detect 头)
            if isinstance(output, (list, tuple)):
                # 如果是多尺度输出，打印每个尺度的形状
                shape_str = str([list(o.shape) for o in output if hasattr(o, 'shape')])
                channels = "Multi-Scale"
            else:
                # 标准层输出
                shape_str = str(list(output.shape))
                channels = output.shape[1]  # 通道数在第1维度 (B, C, H, W)

            print(f"{i:<5} | {module.__class__.__name__:<25} | {str(channels):<15} | {shape_str}")

        return hook


    # 为 model.model.model (Sequential容器) 中的每一层注册钩子
    # 这样就能对应到 YAML 配置文件中的 layer 0, layer 1, ...
    for i, layer in enumerate(model.model.model):
        hooks.append(layer.register_forward_hook(register_hook(i)))

    # 3. 构造一个 640x640 的假输入跑一次前向传播
    try:
        dummy_input = torch.zeros(1, 3, 640, 640)
        # 注意：这里直接调用 model.model 进行推理，绕过预处理，纯测网络结构
        model.model(dummy_input)
    except Exception as e:
        print(f"Error during profiling: {e}")
    finally:
        # 移除钩子，防止影响后续使用
        for h in hooks:
            h.remove()

    print("=" * 90 + "\n")
    # =================================================================

    # 4. 后续操作 (如 fuse)
    # model.fuse()