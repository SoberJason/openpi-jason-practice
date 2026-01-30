import matplotlib.pyplot as plt
import re
import pandas as pd

def parse_training_log(log_file):
    steps = []
    losses = []
    grad_norms = []

    # 正则表达式匹配日志格式
    # 格式参考: Step 10: grad_norm=0.1234, loss=2.3456, ...
    pattern = re.compile(r"Step (\d+):.*grad_norm=([0-9.]+).*loss=([0-9.]+)")

    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                grad_norms.append(float(match.group(2)))
                losses.append(float(match.group(3)))
    
    return pd.DataFrame({'Step': steps, 'Loss': losses, 'GradNorm': grad_norms})

def plot_metrics(df):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 Loss 曲线 (红色)
    color = 'tab:red'
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(df['Step'], df['Loss'], color=color, marker='o', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 绘制 Gradient Norm 曲线 (蓝色) - 用于分析训练稳定性
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Gradient Norm', color=color)  
    ax2.plot(df['Step'], df['GradNorm'], color=color, linestyle='--', label='Grad Norm')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('OpenPI Training Analysis: Loss & Gradient Norm')
    fig.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 请先创建一个 training_log.txt 把终端的输出粘贴进去
    # 如果没有文件，这里用模拟数据演示
    print("正在分析日志...")
    try:
        df = parse_training_log("training_log.txt")
        if not df.empty:
            print(f"成功解析 {len(df)} 条记录")
            print(df.head())
            plot_metrics(df)
        else:
            print("错误：日志文件中未找到匹配的数据，请检查格式。")
    except FileNotFoundError:
        print("请创建一个 'training_log.txt' 文件，并将终端的训练日志复制进去。")