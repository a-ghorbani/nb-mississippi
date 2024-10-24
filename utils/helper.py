import json
import os
import random
import re
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import AutoConfig, AutoModel, AutoTokenizer


def initialize_h2ovl_model(model_path):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    config.vision_config.use_flash_attn = False

    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    return model, tokenizer


def initialize_qwen_model(model_path):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor


def initialize_model(model_path):
    if "h2ovl" in model_path.lower():
        # Initialize H2OVL model
        return initialize_h2ovl_model(model_path)
    elif "qwen" in model_path.lower():
        # Initialize Qwen model
        return initialize_qwen_model(model_path)
    else:
        raise ValueError(f"Unknown model type for path: {model_path}")


def parse_json_response(response):
    """
    Extract JSON object from a response string using regular expressions.
    Assumes the response contains a JSON object that follows the format {"type": ""}.
    If model only returns a single word, then it will be converted to {"type": response}
    """
    try:
        # Use regular expression to find the JSON object in the string
        json_match = re.search(r"\{.*?\}", response)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        elif len(response.strip().split(" ")) == 1:
            return {"type": response.strip()}
        else:
            print(f"Could not find valid JSON in response: {response}")
            return {"type": "other"}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from response: {response}")
        return None


def evaluate_h2ovl_model(model, tokenizer, files, prompt):
    predicted_labels = []
    generation_config = dict(max_new_tokens=500, do_sample=False)
    _prompt = "<image>\n" + prompt
    for image_file, true_label in files:
        response, history = model.chat(
            tokenizer,
            image_file,
            _prompt,
            generation_config,
            history=None,
            return_history=True,
        )

        parsed_response = parse_json_response(response)
        if parsed_response and "type" in parsed_response:
            predicted_type = parsed_response["type"]
        else:
            predicted_type = (
                ""  # Handle cases where parsing fails or JSON is incomplete
            )

        predicted_labels.append(predicted_type)

    return predicted_labels


def evaluate_qwen_model(model, processor, files, prompt):
    predicted_labels = []

    for image_file, true_label in files:
        # Load the image
        image = Image.open(image_file)

        # Create the conversation prompt for Qwen
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preprocess inputs for Qwen
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # Inference
        output_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Extract predicted type from output text
        predicted_type = parse_json_response(output_text[0]).get("type", "")

        predicted_labels.append(predicted_type)

    return predicted_labels


def evaluate_model(model, tokenizer_or_processor, files, prompt):
    actual_labels = [true_label for _, true_label in files]

    if "h2ovl" in model.name_or_path.lower():
        predicted_labels = evaluate_h2ovl_model(
            model, tokenizer_or_processor, files, prompt
        )
    elif "qwen" in model.name_or_path.lower():
        predicted_labels = evaluate_qwen_model(
            model, tokenizer_or_processor, files, prompt
        )
    else:
        raise ValueError(f"Unknown model type for path: {model_path}")

    accuracy = accuracy_score(actual_labels, predicted_labels)

    conf_matrix = confusion_matrix(
        actual_labels, predicted_labels, labels=["invoice", "news-article", "resume"]
    )

    print(f"Accuracy: {accuracy * 100:.2f}%")

    conf_df = pd.DataFrame(
        conf_matrix,
        index=["invoice", "news-article", "resume"],
        columns=["invoice", "news-article", "resume"],
    )

    return accuracy, conf_df, predicted_labels


def load_image_from_file(image_file):
    return Image.open(image_file).convert("RGB")


def plot_images_with_labels(files, images_per_row=5):
    """
    Plots the given images in a grid, with their categories labeled on top.

    Parameters:
    - files: A list of tuples where each tuple contains (file_path, label)
    - images_per_row: Number of images to display per row in the grid (default is 5)
    """
    num_rows = (len(files) + images_per_row - 1) // images_per_row

    fig, axs = plt.subplots(num_rows, images_per_row, figsize=(20, 4 * num_rows))
    axs = axs.flatten()

    for i, (image_file, label) in enumerate(files):
        image = load_image_from_file(image_file)
        axs[i].imshow(image)
        axs[i].axis("off")
        axs[i].text(
            10,
            10,
            label,
            color="red",
            fontsize=15,
            weight="bold",
            backgroundcolor="white",
        )

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


def sample_files_from_directory(base_dir, sampling_dict):
    """
    Sample files by document type from the directory structure.

    Parameters:
    - base_dir (str): The base directory containing subfolders for each document type.
    - sampling_dict (dict): A dictionary where the keys are document types (e.g., 'resume', 'invoice')
                            and the values are the number of samples to take from each type.

    Returns:
    - sampled_files (list of tuples): List of sampled files with no replacement, where each tuple is (file_path, doc_type).
    """
    # Dictionary to hold files grouped by document type (subfolder names)
    doc_type_to_files = defaultdict(list)

    # Traverse the base directory
    for doc_type in os.listdir(base_dir):
        doc_type_path = os.path.join(base_dir, doc_type)

        # Ensure it's a directory representing a document type
        if os.path.isdir(doc_type_path):
            for root, _, files in os.walk(doc_type_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    doc_type_to_files[doc_type].append(file_path)

    # Sample files based on the sampling_dict
    sampled_files = []
    for doc_type, num_samples in sampling_dict.items():
        if doc_type not in doc_type_to_files:
            print(f"Warning: No files found for document type '{doc_type}'")
            continue

        available_files = doc_type_to_files[doc_type]
        if len(available_files) < num_samples:
            print(
                f"Warning: Not enough files for document type '{doc_type}', only {len(available_files)} available."
            )
            num_samples = len(
                available_files
            )  # Sample all available files if not enough

        # Sample without replacement
        sampled_files.extend(
            [
                (file_path, doc_type.replace(" ", "-"))
                for file_path in random.sample(available_files, num_samples)
            ]
        )

    return sampled_files
