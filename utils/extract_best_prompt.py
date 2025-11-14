import os
import csv
import argparse
import re

def find_best_prompt_and_save(directory: str):
    output_dir = "./results/lexica_test/SDXL_Turbo"
    print(f"--- Starting to scan directory: {directory} ---")
    print(f"--- Results will be saved to: {output_dir}/ ---")

    os.makedirs(output_dir, exist_ok=True)
    
    files_processed = 0
    
    for filename in os.listdir(directory)[0:50]:
        if filename.lower().startswith('fuzz_results_') and filename.lower().endswith('.csv'):
            csv_file_path = os.path.join(directory, filename)
            
            max_similarity = -float('inf')
            best_prompt = None
            
            try:
                with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)

                    required_columns = ['clip_similarity', 'full_prompt']
                    
                    if not all(col.strip() in reader.fieldnames for col in required_columns):
                        print(f"[警告] 文件 '{filename}' 缺少必要的列 ('clip_similarity' 或 'full_prompt')，已跳过。")
                        continue

                    for row in reader:
                        try:
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
                        files_processed += 1
                    else:
                        print(f"Warning '{filename}' can not match the expected pattern.")
                else:
                    print(f"Warning '{filename}' did not yield any valid prompts.")

            except Exception as e:
                print(f"Error processing file '{filename}': {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract the 'full prompt' corresponding to the maximum 'clip similarity' from CSV files and save them to corresponding .txt files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "directory", 
        help="Directory containing the CSV files to scan."
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: The provided path '{args.directory}' is not a valid directory.")
    else:
        find_best_prompt_and_save(args.directory)