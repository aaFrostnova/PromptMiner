## Installation

```sh
conda create -n promptinversion python=3.10
conda activate promptinversion
pip install -t requirements.txt
```

## Models

[Salesforce/blip-image-captioning-large · Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-large)

[Qwen/Qwen2-VL-2B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

## Datasets

MS COCO https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval

LAION https://huggingface.co/datasets/laion/laion2B-en-aesthetic

Flickr https://huggingface.co/datasets/Naveengo/flickr8k

Process the data:

```bash
# Firstly, set the paths in data/data.py.
python data/data.py
```



## Run

```bash
# set the paths in run.sh
TARGET_MODEL_PATH="/project/pi_shiqingma_umass_edu/mingzheli/model/sdxl-turbo"

RECORD_DIR="/home/mingzhel_umass_edu/Modifier_fuzz/logs_structured_mcts_fuzzer_sdxl_mscoco"

IMAGE_MODEL_PATH="/project/pi_shiqingma_umass_edu/mingzheli/model/Qwen2-VL-2B-Instruct"

WORK_DIR="./results_sdxl_mscoco"

# run the script
bash run.sh
```

