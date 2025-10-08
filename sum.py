import json

def calculate_averages(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total_cos_sim = 0
    total_lpips = 0
    total_R = 0
    total_P = 0
    total_F1 = 0
    total_ppl = 0
    total_sbert = 0
    count = len(data)
    for image_data, key in zip(data.values(), data.keys()):
        total_cos_sim += image_data['cos_sim']
        total_lpips += image_data['lpips_sim']
        total_R += image_data["R"]
        total_P += image_data["P"]
        total_F1 += image_data["F1"]
        total_sbert += image_data["sbert"]
        total_ppl += image_data["ppl"]

    
    avg_cos_sim = total_cos_sim / count
    avg_lpips = total_lpips / count
    avg_P = total_P / count
    avg_R = total_R / count
    avg_F1 = total_F1 / count
    avg_sbert = total_sbert / count
    avg_ppl = total_ppl / count
    
    print(f"Average cos_sim: {avg_cos_sim:.3f}")
    print(f"Average LPIPS: {avg_lpips:.3f}")
    print(f"Average R: {avg_R:.3f}")
    print(f"Average P: {avg_P:.3f}")
    print(f"Average F1: {avg_F1:.3f}")
    print(f"Average sbert: {avg_sbert:.3f}")
    print(f"Average ppl: {avg_ppl:.3f}")

# 使用文件
# json_file = "/home/mingzhel_umass_edu/inverse/hard-prompts-made-easy/result_flickr/best_texts.json"
json_file = "/project/pi_shiqingma_umass_edu/zwen_umass_edu/baselines/blip_1per/flickr30k/SD35_medium/result.json"
calculate_averages(json_file)