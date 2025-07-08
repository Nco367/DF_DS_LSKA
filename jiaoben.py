import random


def split_numbers_to_files(start, end, random_count, file_a, file_b):
    """
    将[start, end]区间内的数字分配到两个文件：
    - 前一半全部写入 file_a
    - 后一半随机选 random_count 个写入 file_a，剩余写入 file_b

    参数：
        start (int): 起始数字
        end (int): 结束数字
        random_count (int): 从后一半里随机挑选写入 file_a 的数量
        file_a (str): A文件路径
        file_b (str): B文件路径
    """
    # 生成数字列表
    numbers = list(range(start, end + 1))
    half = len(numbers) // 2

    # 前一半、后一半
    first_half = numbers[:half]
    second_half = numbers[half:]

    # 随机挑选
    if random_count > len(second_half):
        raise ValueError("随机数量超过了后一半数字的数量！")

    selected_for_a = random.sample(second_half, random_count)
    remaining_for_b = list(set(second_half) - set(selected_for_a))

    # 写入 A 文件
    with open(file_a, "a") as f_a:
        for i, num in enumerate(first_half):
            if i < len(first_half) - 1:
                f_a.write(f"{num:06d}\n")
            else:
                f_a.write(f"{num:06d}")  # 最后一行无换行

    with open(file_a, "a") as f_a:
        for num in selected_for_a:
            f_a.write(f"\n{num:06d}")

    # 写入 B 文件
    with open(file_b, "a") as f_b:
        for i, num in enumerate(remaining_for_b):
            if i < len(remaining_for_b) - 1:
                f_b.write(f"{num:06d}\n")
            else:
                f_b.write(f"{num:06d}")

    print(f"已完成生成！A文件：{file_a}，B文件：{file_b}")


def main():
    split_numbers_to_files(2640, 3292, 180, "trainA02.txt", "testA02.txt")


if __name__ == "__main__":
    main()
