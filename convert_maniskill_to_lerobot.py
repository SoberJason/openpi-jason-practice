import h5py
import torch
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tqdm
import shutil

# ================= 配置区域 =================
# 输入：生成的包含 RGB 的 .h5 文件
INPUT_H5_PATH = "data/maniskill/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5" 

# 输出：转换后的 LeRobot 数据集存放位置
REPO_ID = "jason/pi05-maniskill-pickcube"
LOCAL_DIR = "data/lerobot_datasets/pi0_maniskill_pickcube"
FPS = 20 # ManiSkill 默认控制频率

# 任务描述 (ManiSkill PickCube-v1 的任务就是抓取红色方块)
TASK_DESCRIPTION = "pick up the red cube"
# ===========================================

def convert_dataset():
    input_path = Path(INPUT_H5_PATH)
    if not input_path.exists():
        print(f"❌ 错误: 找不到输入文件 {input_path}")
        return

    # 如果输出目录存在，先清理掉，防止追加导致数据重复或错误
    if Path(LOCAL_DIR).exists():
        print(f"⚠️  警告: 输出目录 {LOCAL_DIR} 已存在，正在删除以重新生成...")
        shutil.rmtree(LOCAL_DIR)

    print(f"🚀 正在读取原始数据: {input_path}")
    
    # 1. 初始化 LeRobot 数据集
    # 不仅存储图像和动作，还存储机械臂的状态 (qpos + qvel)
    # 注意：这里我们在 features 里不写 "task"，
    # 因为 LeRobot 会自动加上它。如果我们自己写容易参数不对。
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=LOCAL_DIR,
        robot_type="panda",
        # 显式定义特征，确保 OpenPI 能正确识别
        features={
            "observation.images.base_camera": {
                "dtype": "image", 
                "shape": (128, 128, 3), # 根据打印的H5文件结构：128x128
                "names": ["height", "width", "channel"]
            },
            "observation.state": {
                "dtype": "float32", 
                "shape": (18,),
                "names": ["state_dim"]
            },
            "action": {
                "dtype": "float32", 
                "shape": (7,), # 根据打印的H5文件结构：7 dim
                "names": ["action_dim"]
            },
        },
        image_writer_threads=4,
    )

    # 2. 读取 H5 文件并转换
    with h5py.File(input_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
        print(f"📊 发现 {len(traj_keys)} 条轨迹，开始转换...")
        
        for key in tqdm.tqdm(traj_keys, desc="转换进度"):
            traj = f[key]
            
            # --- 提取数据 (基于打印的H5文件结构) ---
            
            # 1. 图像 (uint8 -> Tensor)
            # 路径: /obs/sensor_data/base_camera/rgb
            img_data = traj["obs"]["sensor_data"]["base_camera"]["rgb"][:]

            # 2. 状态 (qpos + qvel)
            # 路径: /obs/agent/qpos, /obs/agent/qvel
            qpos = traj["obs"]["agent"]["qpos"][:]
            qvel = traj["obs"]["agent"]["qvel"][:]
            # 拼接成一个 18维向量
            state_data = np.concatenate([qpos, qvel], axis=-1)

            # 3. 动作
            # 路径: /actions
            action_data = traj["actions"][:]
            
            # --- 长度对齐 ---
            # ManiSkill: Obs (75) = Initial + 74 steps,  Action (74)
            # LeRobot:   要求一一对应 (Obs[i] -> Action[i])
            # 做法: 丢弃最后一帧 Obs (它是执行完最后一个动作后的结果，没有下一个动作了)
            n_actions = action_data.shape[0]
            
            # --- 写入 LeRobot ---
            for i in range(n_actions):
                frame_dict = {
                    "observation.images.base_camera": torch.from_numpy(img_data[i]),
                    "observation.state": torch.from_numpy(state_data[i]).float(),
                    "action": torch.from_numpy(action_data[i]).float(),
                    # 关键修改：虽然上面 features 没写，但校验器要这个，我们必须给！
                    "task": TASK_DESCRIPTION
                }
                dataset.add_frame(frame_dict)
            
            # 标记一条轨迹结束
            dataset.save_episode()
     
    # 3. 整合并保存统计信息       
    print("💾 正在整合数据集...")
    dataset.consolidate()
    print(f"\n✅ 转换成功！数据集位置: {LOCAL_DIR}")

if __name__ == "__main__":
    convert_dataset()