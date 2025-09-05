import os
import csv
import argparse
import re

def find_best_prompt_and_save(directory: str):
    """
    扫描目录中的所有CSV文件，为每个文件找到最大 'clip similarity' 首次出现时
    对应的 'full prompt'，并将其写入 'results_sdxl_txt' 文件夹中对应的 .txt 文件。
    """
    output_dir = "results_sdxl_txt_mscoco"
    print(f"--- 开始扫描文件夹: {directory} ---")
    print(f"--- 结果将被保存到: {output_dir}/ ---")

    # 步骤 1: 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    files_processed = 0
    
    for filename in os.listdir(directory):
        if filename.lower().startswith('fuzz_results_') and filename.lower().endswith('.csv'):
            csv_file_path = os.path.join(directory, filename)
            
            # 为了清晰，我们将变量名改为 max_similarity
            max_similarity = -float('inf')
            best_prompt = None
            
            try:
                with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    
                    # --- 核心修改 1: 更新需要检查的列名 ---
                    required_columns = ['clip_similarity', 'full_prompt']
                    
                    if not all(col.strip() in reader.fieldnames for col in required_columns):
                        print(f"[警告] 文件 '{filename}' 缺少必要的列 ('clip_similarity' 或 'full_prompt')，已跳过。")
                        continue

                    # 单次遍历查找最佳 prompt
                    for row in reader:
                        try:
                            # --- 核心修改 2: 从 'clip similarity' 列获取分数 ---
                            current_similarity_str = row.get('clip_similarity', '-1')
                            current_similarity = float(current_similarity_str)

                            if current_similarity > max_similarity:
                                max_similarity = current_similarity
                                best_prompt = row.get('full_prompt')
                        except (ValueError, TypeError):
                            continue
                
                if best_prompt is not None:
                    match = re.search(r'fuzz_results_(\w+)\.csv', filename, re.IGNORECASE)
                    if match:
                        base_name = match.group(1)
                        txt_filename = f"{base_name}.txt"
                        txt_file_path = os.path.join(output_dir, txt_filename)

                        with open(txt_file_path, 'w', encoding='utf-8') as txtfile:
                            txtfile.write(best_prompt)
                        
                        # --- 核心修改 3: 更新打印信息 ---
                        print(f"[成功] 文件: '{filename}' | 最大 Clip Similarity: {max_similarity:.4f} | Prompt 已写入 '{txt_file_path}'")
                        files_processed += 1
                    else:
                        print(f"[警告] 文件名 '{filename}' 格式不匹配，无法生成对应的.txt文件。")
                else:
                    print(f"[警告] 未在 '{filename}' 中找到任何有效的数据行。")

            except Exception as e:
                print(f"[错误] 处理文件 '{filename}' 时出错: {e}")

    print(f"\n--- 操作完成 ---")
    print(f"共为 {files_processed} 个 CSV 文件生成了对应的 .txt 文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        # --- 核心修改 4: 更新脚本描述 ---
        description="从CSV文件中提取最大'clip similarity'对应的'full prompt'，并保存到对应的.txt文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "directory", 
        help="要扫描的包含CSV文件的目标文件夹路径。"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"错误：提供的路径 '{args.directory}' 不是一个有效的文件夹。")
    else:
        find_best_prompt_and_save(args.directory)