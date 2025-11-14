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

We have a dataset example in ./data;
You can customize your own target image according to the format of the dataset.



## Run

```bash
# set the paths in run.sh
TARGET_MODEL_PATH="path to your model"

RECORD_DIR="path for Phase II results"

IMAGE_MODEL_PATH="path to Qwen2-vl"

WORK_DIR="path for Phase I results"

# run the script
bash run.sh

# extratc the results
python utils/extract_best_prompt.py RECORD_DIR
# you can modify line 7 in utils/extract_best_prompt.py to change the output_dir
```
## evaluation

```bash
python eval.py
```