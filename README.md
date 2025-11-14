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
TARGET_MODEL_PATH="path to your model"

RECORD_DIR="path for Phase II results"

IMAGE_MODEL_PATH="path to Qwen2-vl"

WORK_DIR="path for Phase I results"

# run the script
bash run.sh
```
## evaluation

modifty the path in eval.py 

```python
# set the paths in run.sh
# this path is where you store all the txt files with generated prompts(000.txt, 001.txt), and one prompt per line
RESULTS_DIR = Path("/home/mingzhel_umass_edu/Modifier_fuzz/results_sdxl_txt_mscoco_5000pre_lexica_baseonly")
# this path is where you store the lexica dataset, you should have prompts.json and images(000.png, 001.png) in this folder
LEXICA_DIR = Path("/home/mingzhel_umass_edu/inverse/LatentTracer/data/lexica")
# this is where the final results will be saved
RESULT_FILE = RESULTS_DIR / "result.json"
# this is the path to the prompts.json file in the lexica dataset(datasets can be changed to other datasets as well)
PROMPTS_FILE = LEXICA_DIR / "prompts.json"
# modify this path to your model path
MODEL_ID = "/project/pi_shiqingma_umass_edu/mingzheli/model/sdxl-turbo"
```
and then run it
```bash
python eval.py
```