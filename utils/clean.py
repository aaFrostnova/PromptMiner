import os
import csv
import argparse

def clean_header_only_csv(directory: str, is_dry_run: bool):
    """
    扫描指定目录，删除所有只包含表头的CSV文件。

    Args:
        directory (str): 要扫描的目标文件夹路径。
        is_dry_run (bool): 如果为True，则只打印将要删除的文件，而不实际删除。
    """
    print(f"--- 开始扫描文件夹: {directory} ---")
    if is_dry_run:
        print("--- 当前为演习模式 (Dry Run)，不会实际删除任何文件 ---")

    # 计数器
    files_found = 0
    files_to_delete = 0
    files_deleted = 0
    print("begin")
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否为CSV文件
        if filename.lower().endswith('.csv'):
            files_found += 1
            file_path = os.path.join(directory, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    
                    # 尝试读取第一行 (表头)
                    try:
                        next(reader)
                    except StopIteration:
                        # 文件是完全空的 (0行)，跳过
                        print(f"[跳过] '{filename}' 是空文件。")
                        continue

                    # 尝试读取第二行 (数据)
                    try:
                        next(reader)
                        # 如果能成功读取第二行，说明文件有内容，跳过
                        # print(f"[保留] '{filename}' 包含数据。") # 可以取消注释以显示更详细信息
                        continue
                    except StopIteration:
                        # 如果在这里触发 StopIteration，说明文件只有一行
                        files_to_delete += 1
                        print(f"[目标] '{filename}' 只包含表头，将被删除。")
                        
                        if not is_dry_run:
                            os.remove(file_path)
                            files_deleted += 1
                            print(f"    └── [已删除] '{filename}'")

            except Exception as e:
                print(f"[错误] 处理文件 '{filename}' 时出错: {e}")
    
    print("\n--- 扫描完成 ---")
    print(f"共找到 {files_found} 个 CSV 文件。")
    if is_dry_run:
        print(f"在演习模式下，有 {files_to_delete} 个文件被识别为待删除。")
    else:
        print(f"共删除了 {files_deleted} 个只有表头的文件。")


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="删除一个文件夹中所有只有表头没有内容的CSV文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "directory", 
        help="要扫描的目标文件夹的路径。"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="演习模式：只列出将要删除的文件，不执行任何删除操作。"
    )
    
    args = parser.parse_args()
    
    # 检查路径是否存在且是一个目录
    if not os.path.isdir(args.directory):
        print(f"错误：提供的路径 '{args.directory}' 不是一个有效的文件夹。")
    else:
        clean_header_only_csv(args.directory, args.dry_run)