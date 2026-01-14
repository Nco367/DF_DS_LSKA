import os
import re
import pandas as pd
import matplotlib.pyplot as plt


# Step 1: 定义读取文件夹中所有日志文件的函数
def read_log_files(log_folder, start_epoch, end_epoch):
    log_data = []  # 用于存储所有文件中的数据
    file_pattern = re.compile(r'epoch_(\d+)_test_log.txt')  # 匹配文件名格式

    # 遍历文件夹中的所有文件
    for filename in os.listdir(log_folder):
        # 检查文件名是否符合 pattern
        match = file_pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))  # 提取 epoch 数字

            if start_epoch <= epoch_num <= end_epoch:
                file_path = os.path.join(log_folder, filename)
                with open(file_path, 'r') as file:
                    lines = file.readlines()  # 读取所有行
                    if lines:  # 检查文件是否非空
                        last_line = lines[-1]  # 获取最后一行
                        # 假设最后一行格式为 "2024-12-24 14:21:14,625 : Test time 00h 12m 28s Epoch 1 TEST FINISH Avg dis: 0.029606662066039926"
                        if 'Avg dis:' in last_line:
                            # 提取 "Avg dis:" 后的数值
                            avg_dis_value = float(last_line.split('Avg dis:')[-1].strip())
                            log_data.append((epoch_num, avg_dis_value))
    return log_data


# Step 2: 可视化数据
def plot_loss(log_data):
    # 将数据转换为 Pandas DataFrame
    df = pd.DataFrame(log_data, columns=['Epoch', 'Avg dis'])

    # 根据 Epoch 对数据进行排序
    df = df.sort_values(by='Epoch')

    # 绘制折线图、
    plt.plot(df['Epoch'], df['Avg dis'], label='Avg dis')
    plt.xlabel('Epoch')
    plt.ylabel('Avg dis')
    plt.title('Training Avg dis over Epochs')
    plt.legend()
    plt.show()


# Step 3: 主函数
if __name__ == "__main__":
    log_folder = '/media/q/SSD2T/1linux/Linemod3_log/20250921_014636(ds lska)/'  # 设置你的日志文件夹路径
    start_epoch = 0  # 指定起始 epoch
    end_epoch = 191# 指定结束 epoch
    log_data = read_log_files(log_folder, start_epoch, end_epoch)

    plot_loss(log_data)
