import os
import sys
import dataclasses
import numpy as np
import jax
import logging

# 强制使用 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from openpi.training import config
from openpi.policies import policy_config 

def main():
    logging.basicConfig(level=logging.INFO)
    print("🚀 正在初始化 JAX (GPU 1)...")
    
    # 1. 加载配置
    config_name = "pi0_maniskill_pickcube"
    print(f"📖 加载配置: {config_name}")
    try:
        train_config = config.get_config(config_name)
        train_config = dataclasses.replace(train_config, exp_name="pi0_maniskill_lora")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return

    checkpoint_dir = train_config.checkpoint_dir
    print(f"📂 Checkpoint 目录: {checkpoint_dir}")

    if not checkpoint_dir.exists():
        print("❌ 目录不存在")
        return

    steps = [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not steps:
        print(f"❌ 未找到 Step 目录")
        return
    
    latest_dir = max(steps, key=lambda p: int(p.name))
    print(f"✨ 自动定位到最新 Checkpoint: {latest_dir.name}")
    
    # 2. 加载 Policy
    print("⚖️ 正在加载 Policy...")
    try:
        policy = policy_config.create_trained_policy(
            train_config, 
            checkpoint_dir=latest_dir
        )
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("✅ 模型加载成功！")

    # ⚠️⬇️ 暴力修复: 直接修改 Policy 实例内部的 Transform
    print("🔧 正在检查 Policy 内部的 Output Transforms...")
    
    def recursive_patch_transforms(transform_tree):
        # 如果是 Normalizer (通常包含 mean/std)
        if hasattr(transform_tree, "mean") and hasattr(transform_tree, "std"):
            mean_shape = transform_tree.mean.shape
            if mean_shape and mean_shape[0] == 18:
                print(f"   🔪 发现 18维 Stats，正在截断为 7维...")
                new_transform = dataclasses.replace(
                    transform_tree,
                    mean=transform_tree.mean[:7],
                    std=transform_tree.std[:7]
                )
                return new_transform
        
        # 递归遍历 Group 或其他组合 Transform
        if hasattr(transform_tree, "transforms"): # 可能是 Group
            new_transforms = []
            for t in transform_tree.transforms:
                new_transforms.append(recursive_patch_transforms(t))
            return dataclasses.replace(transform_tree, transforms=tuple(new_transforms))
            
        # 递归遍历 dict (有些实现是 dict 结构)
        if isinstance(transform_tree, dict):
            for k, v in transform_tree.items():
                transform_tree[k] = recursive_patch_transforms(v)
            return transform_tree
            
        return transform_tree

    # 尝试访问 _output_transform (这是 Policy 用来反归一化的私有属性)
    if hasattr(policy, "_output_transform"):
        try:
            print("   -> 正在扫描 _output_transform")
            
            def deep_modify_arrays(obj, visited=None):
                if visited is None: visited = set()
                if id(obj) in visited: return
                visited.add(id(obj))
                
                if dataclasses.is_dataclass(obj):
                    for field in dataclasses.fields(obj):
                        val = getattr(obj, field.name)
                        if isinstance(val, (np.ndarray, jax.Array)):
                            if val.shape == (18,):
                                print(f"      🎯 命中属性 '{field.name}' (18,) -> 强制截断 (7,)")
                                setattr(obj, field.name, val[:7])
                        else:
                            deep_modify_arrays(val, visited)
                
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        deep_modify_arrays(item, visited)
                
                elif isinstance(obj, dict):
                    for val in obj.values():
                        deep_modify_arrays(val, visited)
                        
                if hasattr(obj, "__dict__"):
                    deep_modify_arrays(obj.__dict__, visited)
            
            deep_modify_arrays(policy)
            
        except Exception as e:
            print(f"⚠️ 无法自动修复 Transforms: {e}")
    else:
        print("⚠️ Policy 没有 _output_transform 属性，可能不需要修复或结构不同。")


    # 3. 构造测试输入
    print("🔧 构造测试输入...")
    dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
    observation = {
        "image": {
            "base_0_rgb": dummy_img,
            "left_wrist_0_rgb": dummy_img,
            "right_wrist_0_rgb": dummy_img,
        },
        "image_mask": {
            "base_0_rgb": np.array(True), 
            "left_wrist_0_rgb": np.array(True),
            "right_wrist_0_rgb": np.array(True),
        },
        "state": np.zeros((7,), dtype=np.float32)
    }

    # 4. 运行推理
    print("🏃 开始推理...")
    try:
        result = policy.infer(observation)
        print("\n🎉 推理成功！")
        
        if hasattr(result, "actions"):
             action = result.actions
        else:
             action = result

        if hasattr(action, 'shape'):
             print(f"📊 输出动作形状: {action.shape}")
             flat_action = action.flatten()
             print(f"🔢 动作值示例: {flat_action[:10]}")
             
             if np.allclose(flat_action, 0):
                 print("⚠️  警告: 全零动作")
             else:
                 print("✅ 动作值正常")
        else:
             print(f"输出结果: {action}")

    except Exception as e:
        print(f"❌ 推理出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()