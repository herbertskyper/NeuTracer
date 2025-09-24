import json
import sys
import uuid
import time
from datetime import datetime


def generate_new_func_call_id(original_id):
    """基于原始ID生成新的func_call_id，在末尾加上字母n"""
    return f"{original_id}n"


def process_file(input_file, target_function, new_function, output_file=None):
    """
    处理文件，在指定函数后添加新的记录

    Args:
        input_file: 输入文件路径
        target_function: 要查找的目标函数名
        new_function: 要添加的新函数名
        output_file: 输出文件路径，如果为None则覆盖原文件
    """
    if output_file is None:
        output_file = input_file

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            new_lines.append(line)

            try:
                # 解析JSON行
                data = json.loads(line)

                # 检查是否是目标函数
                if data.get("function") == target_function:
                    # 创建新记录，复制所有字段
                    new_record = data.copy()

                    # 基于原始ID生成新ID（加字母n）
                    original_id = data.get("func_call_id", "")
                    new_record["func_call_id"] = generate_new_func_call_id(original_id)
                    new_record["function"] = new_function

                    # 添加新记录到下一行
                    new_lines.append(json.dumps(new_record, separators=(",", ":")))

                    print(f"找到目标函数 '{target_function}'")
                    print(f"原始ID: {original_id}")
                    print(f"新ID: {new_record['func_call_id']}")
                    print(f"添加了新函数 '{new_function}' 的记录")
                    print("-" * 50)

            except json.JSONDecodeError:
                # 如果不是有效的JSON，跳过
                continue

        # 写入结果
        with open(output_file, "w", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")

        print(f"处理完成！已将结果写入 {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"处理过程中出现错误：{e}")


def main():
    if len(sys.argv) < 4:
        print("使用方法:")
        print("python script.py <输入文件> <目标函数名> <新函数名> [输出文件]")
        print("\n示例:")
        print(
            "python script.py data.txt 'torch.optim.adadelta.Adadelta.step' 'torch.custom.function'"
        )
        print(
            "python script.py data.txt 'torch.optim.adadelta.Adadelta.step' 'torch.custom.function' output.txt"
        )
        return

    input_file = sys.argv[1]
    target_function = sys.argv[2]
    new_function = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else None

    process_file(input_file, target_function, new_function, output_file)


if __name__ == "__main__":
    main()
