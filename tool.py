def add_prefix_to_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in lines:
            # 在每一行前加 "00"
            outfile.write("00" + line)

# 使用示例
input_file = '/media/q/HDD3T_1.5TB/2linux/datebase/DenseFusiondatasets/linemod2/Linemod_preprocessed/data/01/train.txt'  # 输入文件路径
output_file = '/media/q/HDD3T_1.5TB/2linux/datebase/DenseFusiondatasets/linemod2/Linemod_preprocessed/data/01/train2.txt'  # 输出文件路径
add_prefix_to_lines(input_file, output_file)
