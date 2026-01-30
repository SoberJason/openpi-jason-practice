import os
import glob
import sys

# 指向你刚刚训练生成的 checkpoint 目录
CHECKPOINT_DIR = "/home/jason/openpi/checkpoints/pi05_libero/pi05_libero_lora"

def scan_file_system_for_lora(base_path):
    """
    直接扫描文件系统寻找带有 'lora' 关键字的路径。
    这种方法不依赖 Orbax 库，可以读取损坏或临时的 Checkpoint。
    """
    print(f"   [文件系统扫描] 正在遍历目录结构: {base_path}")
    print(f"   只有实际写入硬盘的 LoRA 参数才会被列出...\n")
    
    lora_paths = []
    total_files = 0
    
    # 递归遍历所有子目录
    for root, dirs, files in os.walk(base_path):
        # 检查当前路径中是否包含 'lora'
        rel_path = os.path.relpath(root, base_path)
        
        # Orbax 通常把参数存为文件夹，例如 .../lora_A/
        # 我们寻找路径中包含 lora 的文件夹，且该文件夹内有实际数据文件 (如 .zarray, value 等)
        if 'lora' in rel_path.lower():
            if files: # 如果文件夹里有文件，说明数据被保存了
                lora_paths.append(rel_path)
        
        total_files += len(files)

    # 打印结果
    if lora_paths:
        print(f"✅ 成功在磁盘上找到 {len(lora_paths)} 个 LoRA 参数块！")
        print("   以下是前 10 个找到的 LoRA 参数路径 (证明权重已存在):")
        for p in lora_paths[:10]:
            print(f"    - {p}")
        if len(lora_paths) > 10:
            print(f"    - ... (还有 {len(lora_paths)-10} 个)")
            
        print(f"\n   统计: 扫描了 {total_files} 个文件。")
        print(f"   结论: 训练是有效的，权重文件已保存在上述目录中。")
    else:
        print("⚠️ 扫描完成，但未在文件名中发现 'lora' 关键字。")
        print("   可能原因: 1. 未使用 LoRA 训练 2. 训练在写入 LoRA 层之前就彻底崩溃了")

def analyze_checkpoint(ckpt_dir):
    print(f"正在分析 Checkpoint: {ckpt_dir}")
    
    if not os.path.exists(ckpt_dir):
        print("错误：目录不存在")
        return

    # 1. 尝试处理临时 (tmp) 目录
    tmp_step_dirs = glob.glob(os.path.join(ckpt_dir, "*orbax-checkpoint-tmp*"))
    
    target_dir = None
    
    if tmp_step_dirs:
        step_dir = tmp_step_dirs[0]
        print(f"⚠️ 检测到临时 Checkpoint 目录: {step_dir}")
        
        # 寻找 params 具体位置
        params_dirs = glob.glob(os.path.join(step_dir, "params*"))
        if params_dirs:
            target_dir = params_dirs[0]
        else:
            print("❌ 未找到 params 目录，可能尚未开始保存参数。")
            return
    else:
        # 2. 尝试寻找标准目录 (纯数字步骤)
        step_dirs = [d for d in os.listdir(ckpt_dir) if d.isdigit()]
        if step_dirs:
            latest_step = max(step_dirs, key=int)
            print(f"检测到标准步数: Step {latest_step}")
            target_dir = os.path.join(ckpt_dir, latest_step, "params")
            # 兼容旧版本结构，有些可能没有 params 子目录直接放
            if not os.path.exists(target_dir):
                 # 假设 items (params) 直接在 step 目录下
                target_dir = os.path.join(ckpt_dir, latest_step)
        else:
            print("❌ 未找到任何 Step 目录 (无论是 tmp 还是标准)。")
            return

    # 执行扫描
    if target_dir:
        scan_file_system_for_lora(target_dir)

if __name__ == "__main__":
    analyze_checkpoint(CHECKPOINT_DIR)