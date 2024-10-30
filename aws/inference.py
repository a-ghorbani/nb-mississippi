# inference.py
import json
import re
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from PIL import Image
import base64
import io

def initialize_h2ovl_model(model_path):
    device = 'cuda' # TODO: check if CPU can be used
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Disable flash attention if it's not supported by the GPU
    config.vision_config.use_flash_attn = False

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def classify_document(model, tokenizer, image_data):
    prompt = "<image>\nExtract the type of the image, categorizing it as 'invoice', 'resume', or 'news-article'. Type:"

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    except Exception as e:
        print(f"Error loading image: {e}")
        return "error-load" # TODO: add error handling

    try:
        response, history = model.chat(tokenizer, image, prompt, generation_config=None, history=None, return_history=True)
    except Exception as e:
        print(f"Error during model chat: {e}")
        return "error-chat" # TODO: add error handling

    parsed_response = parse_json_response(response)
    return parsed_response.get("type", "error-parsing") # TODO: add error handling

def parse_json_response(response):
    try:
        json_match = re.search(r'\{.*?\}', response)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        elif len(response.strip().split(" ")) == 1:
            return {"type": response.strip()}
        else:
            print(f"Could not find valid JSON in response: {response}")
            return {"type": 'error-parsing'} # TODO: add error handling
    except json.JSONDecodeError:
        print(f"Error decoding JSON from response: {response}")
        return {"type": 'error-parsing'} # TODO: add error handling

# Initialize the model and tokenizer globally for reuse
model, tokenizer = initialize_h2ovl_model('h2oai/h2ovl-mississippi-800m')

def handler(event, context):
    image_data = event.get("image_data", "")
    if image_data:
        try:
            predicted_type = classify_document(model, tokenizer, image_data)
            return {"statusCode": 200, "body": json.dumps({"type": predicted_type})}
        except Exception as e:
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
    else:
        return {"statusCode": 400, "body": json.dumps({"error": "No image data provided"})}
