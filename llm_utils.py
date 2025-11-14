import torch
from fastchat.model import load_model
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration

@torch.inference_mode()
def create_model(args, model_path):
    model, tokenizer = load_model(
        model_path,
        args.device,
        args.num_gpus,
        args.dtype,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        revision=args.revision,
        # debug=args.debug,
    )
    return model, tokenizer


def create_model_and_tok(args, model_path, target=False):
    openai_model_list = ['gpt-4o-mini', 'gpt-4o-2024-05-13','meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Llama-2-70b-chat-hf','mistralai/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mixtral-8x22B-Instruct-v0.1', 'gpt-3.5-turbo-1106','gpt-3.5-turbo-0613', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301']
   
    if model_path in openai_model_list:
        MODEL = model_path
        TOK = None
    else:
        MODEL, TOK = create_model(args, model_path)

    return MODEL, TOK


def prepare_model_and_tok(args, target=False):
    if type(args.model_path) == str:
        MODEL, TOK = create_model_and_tok(args, args.model_path, target=target)
    elif type(args.model_path) == list:
        MODEL, TOK = [], []
        for model_path in args.model_path:
            model, tok = create_model_and_tok(args, model_path)
            MODEL.append(model)
            TOK.append(tok)
    else:
        raise NotImplementedError
    return MODEL, TOK

def prepare_model_and_tok_vl(args):
    
    if type(args.image_model_path) == str:
        PROCESSOR = AutoProcessor.from_pretrained(args.image_model_path)
        if 'Qwen2.5' in args.image_model_path:
            VL_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.image_model_path
            ).to(args.device)
        else:
            VL_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
                args.image_model_path
            ).to(args.device)
    elif type(args.image_model_path) == list:
        MODEL, TOK = [], []
        for image_model_path in args.image_model_path:
            processor = AutoProcessor.from_pretrained(image_model_path)
            if 'Qwen2.5' in image_model_path:
                vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    image_model_path
                ).to(args.device)
            else:
                vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    image_model_path
                ).to(args.device)
            PROCESSOR.append(processor)
            VL_MODEL.append(vl_model)
    else:
        raise NotImplementedError

    return PROCESSOR, VL_MODEL