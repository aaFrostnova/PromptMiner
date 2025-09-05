import os
import argparse

def trim_csv_files(directory: str, keep_limit: int, is_dry_run: bool):
    """
    扫描指定目录，按字母顺序对CSV文件排序，并删除第 N 个之后的所有文件。

    Args:
        directory (str): 要扫描的目标文件夹路径。
        keep_limit (int): 要保留的文件数量（例如30）。
        is_dry_run (bool): 如果为True，则只打印将要删除的文件，而不实际删除。
    """
    print(f"--- 开始扫描文件夹: {directory} ---")
    if is_dry_run:
        print("--- 当前为演习模式 (Dry Run)，不会实际删除任何文件 ---")

    # 步骤 1: 查找所有CSV文件并获取其完整路径
    try:
        all_files = os.listdir(directory)
        csv_files = [
            os.path.join(directory, f)
            for f in all_files
            if f.lower().endswith('.csv') and os.path.isfile(os.path.join(directory, f))
        ]
    except FileNotFoundError:
        print(f"错误：文件夹 '{directory}' 不存在。")
        return
    except Exception as e:
        print(f"错误：读取文件夹时出错: {e}")
        return

    # 步骤 2: 按字母顺序对文件列表进行排序
    csv_files.sort()
    
    total_found = len(csv_files)
    print(f"共找到 {total_found} 个 CSV 文件，并已按字母顺序排序。")

    # 步骤 3: 检查文件总数是否超过限制
    if total_found <= keep_limit:
        print(f"文件总数未超过 {keep_limit} 个，无需删除任何文件。")
        print("--- 操作完成 ---")
        return

    # 步骤 4: 确定要删除的文件列表
    # Python 列表索引从0开始，所以要保留30个，即保留索引 0-29 的文件
    # 从索引 30 (第31个文件) 开始删除
    files_to_delete = csv_files[keep_limit:]
    
    print(f"将保留前 {keep_limit} 个文件，准备删除其余 {len(files_to_delete)} 个文件...")

    # 步骤 5: 执行删除或打印操作
    files_deleted_count = 0
    for file_path in files_to_delete:
        try:
            if is_dry_run:
                print(f"[待删除] {os.path.basename(file_path)}")
            else:
                os.remove(file_path)
                print(f"[已删除] {os.path.basename(file_path)}")
            files_deleted_count += 1
        except Exception as e:
            print(f"[错误] 删除文件 '{file_path}' 时出错: {e}")

    print("\n--- 操作完成 ---")
    if is_dry_run:
        print(f"在演习模式下，有 {files_deleted_count} 个文件被识别为待删除。")
    else:
        print(f"成功删除了 {files_deleted_count} 个文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="按字母顺序排序并删除一个文件夹中第 N 个之后的所有CSV文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "directory", 
        help="要扫描的目标文件夹的路径。"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="要保留的CSV文件数量。默认为30。"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="演习模式：只列出将要删除的文件，不执行任何删除操作。"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"错误：提供的路径 '{args.directory}' 不是一个有效的文件夹。")
    else:
        trim_csv_files(args.directory, args.limit, args.dry_run)