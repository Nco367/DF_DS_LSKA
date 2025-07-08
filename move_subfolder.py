import os
import shutil
import concurrent.futures
import threading
from queue import Queue
# 添加进度条（需安装tqdm）
from tqdm import tqdm

# 日志队列用于线程安全的输出
log_queue = Queue()
# 线程锁用于处理文件重命名冲突
rename_lock = threading.Lock()

def move_item(content_path, dest_path, parent_dir):
    """
    移动单个文件/目录的线程任务
    :param content_path: 源路径
    :param dest_path: 目标路径
    :param parent_dir: 父目录（用于日志显示）
    :return: (操作结果, 源名称, 目标名称)
    """
    global rename_lock
    
    content_name = os.path.basename(content_path)
    original_dest = dest_path
    
    try:
        # 检查目标是否存在（需要加锁避免竞争）
        with rename_lock:
            if os.path.exists(dest_path):
                # 自动重命名策略
                base, ext = os.path.splitext(content_name)
                counter = 1
                while os.path.exists(dest_path):
                    new_name = f"{base}_moved_{counter}{ext}"
                    dest_path = os.path.join(parent_dir, new_name)
                    counter += 1
                log_queue.put(f"重命名: {content_name} -> {os.path.basename(dest_path)}")

        # 执行移动操作
        shutil.move(content_path, dest_path)
        return (True, content_name, os.path.basename(dest_path))
    except Exception as e:
        return (False, content_name, str(e))

def process_subfolder(item_path, parent_dir):
    """
    处理单个子文件夹的线程任务
    :param item_path: 子文件夹完整路径
    :param parent_dir: 父目录路径
    """
    item_name = os.path.basename(item_path)
    log_queue.put(f"\n处理子文件夹: {item_name}")

    # 创建线程池（根据文件数量自动调整）
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()*2) as executor:
        futures = []
                
        # 提交移动任务
        for content_name in os.listdir(item_path):
            content_path = os.path.join(item_path, content_name)
            dest_path = os.path.join(parent_dir, content_name)
            futures.append(executor.submit(move_item, content_path, dest_path))

        # 添加进度条（关键修改位置）
        with tqdm(
            total=len(futures),
            desc=f"正在移动 {item_name}",
            unit="文件",
            position=int(item_name.split("_")[-1]) if "_" in item_name else 0  # 可选：根据文件夹序号定位进度条
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                success, result = future.result()
                if success:
                    log_queue.put(f"移动成功: {result}")
                else:
                    log_queue.put(f"移动失败: {result}")
                pbar.update(1)  # 更新进度
    # 尝试删除空文件夹
    try:
        os.rmdir(item_path)
        log_queue.put(f"成功删除空文件夹: {item_name}")
    except OSError as e:
        log_queue.put(f"删除失败: {item_name} - {str(e)}")


def log_worker():
    """日志输出线程"""
    while True:
        message = log_queue.get()
        if message is None:  # 终止信号
            break
        print(message)
        log_queue.task_done()

def move_subfolder_contents_to_parent(parent_dir):
    """
    主处理函数
    :param parent_dir: 要处理的父目录
    """
    # 验证目录有效性
    if not os.path.isdir(parent_dir):
        raise ValueError(f"路径不是目录或不存在: {parent_dir}")

    # 启动日志线程
    logging_thread = threading.Thread(target=log_worker)
    logging_thread.start()

    try:
        # 创建文件夹处理线程池
        with concurrent.futures.ThreadPoolExecutor(max(1, os.cpu_count()//2)) as folder_executor:
            folder_futures = []
            # 遍历所有子文件夹并提交任务
            for item_name in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item_name)
                if os.path.isdir(item_path):
                    folder_futures.append(
                        folder_executor.submit(
                            process_subfolder,
                            item_path,
                            parent_dir
                        )
                    )

            # 等待所有子文件夹处理完成
            for future in concurrent.futures.as_completed(folder_futures):
                future.result()  # 捕获可能异常

    finally:
        # 停止日志线程
        log_queue.put(None)
        logging_thread.join()

if __name__ == "__main__":
    import argparse
    
    """
    执行逻辑：
    1. 通过命令行参数获取父目录路径
    2. 启动日志输出线程
    3. 遍历父目录下的所有子文件夹
    4. 对每个子文件夹：
       a. 创建文件移动线程池（8线程）
       b. 并行移动所有内容到父目录
       c. 处理文件命名冲突
       d. 删除空文件夹
    5. 所有操作完成后输出总结报告
    """
    
    parser = argparse.ArgumentParser(description='并行移动子文件夹内容到父目录')
    parser.add_argument('parent_dir', default='/media/q/新加卷/BaiduYun/YCB_Video_Dataset/data_syn/',type=str, help='父目录路径')
    args = parser.parse_args()

    try:
        move_subfolder_contents_to_parent(args.parent_dir)
        print("\n操作完成！")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")