"""
Step 3: Collect layer-9 last-token activations from the frozen SFT model.

Step 5: Collect Activations from SFT Model
Extract hidden states at the answer position for SAE training.
"""


import os  # environment variables (e.g., CUDA device order)
import json  # read json/jsonl data
import torch  # PyTorch
import numpy as np  # numerics
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace models
from pathlib import Path  # path handling
import argparse  # command-line arguments
from tqdm import tqdm  # progress bar
import pickle  # save Python objects


import _bootstrap


def collect_activations(model, tokenizer, qa_file, layer_idx=-1, max_samples=None):
    """
    Collect activation vectors from a given layer of the model.
    
    Uses the plain QA format (no biography context),
    so the model must answer from its internal knowledge.
    
    Args:
        model: fine-tuned language model
        tokenizer: matching tokenizer
        qa_file: QA data file
        layer_idx: layer to extract (-1 = last layer)
        max_samples: maximum number of samples
    """

    # device of the model (GPU / CPU)
    device = next(model.parameters()).device

    # ---------------------------
    # load QA data
    # ---------------------------
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]

    # truncate if max_samples is set
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]

    # collected activations
    activations = []

    # eval mode
    model.eval()

    # no gradients (saves memory)
    with torch.no_grad():

        # iterate over QA samples
        for qa in tqdm(qa_pairs, desc="Collecting activations"):

            # ---------------------------
            # build the input prompt
            # ---------------------------
            # question only, no biography
            # tests whether the model has memorized the fact
            prompt = f"Q: {qa['question']}\nA:"

            # tokenize
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            # ---------------------------
            # forward pass
            # ---------------------------
            # output_hidden_states=True returns the hidden states of all layers
            outputs = model(**inputs, output_hidden_states=True)

            # ---------------------------
            # hidden states of the chosen layer
            # ---------------------------
            # hidden_states is a tuple:
            # (embedding, layer1, layer2, ... layerN)

            hidden_states = outputs.hidden_states[layer_idx]  # shape: [1, seq_len, d_model]

            # position of the last prompt token,
            # i.e., where the model starts generating the answer
            last_position = inputs['input_ids'].shape[1] - 1

            # hidden state at that position
            # shape: [d_model]
            h = hidden_states[0, last_position, :].cpu().numpy()

            # ---------------------------
            # answer tokens
            # ---------------------------
            # tokenize the answer
            answer_tokens = tokenizer.encode(
                f" {qa['answer']}",
                add_special_tokens=False
            )

            # ---------------------------
            # record
            # ---------------------------
            activations.append({
                'h': h,  # hidden state vector
                'concept_idx': qa['concept_idx'],  # concept id, e.g., 0
                'concept': qa.get('concept'),  # concept name, e.g., "dog"
                'answer': qa['answer'],  # answer text
                'answer_tokens': answer_tokens,  # answer tokens
                'property_idx': qa['property_idx'],  # property ID
                'property': qa.get('property'),  # property name
                'question': qa['question'],  # question text
                'synonym_idx': qa['synonym_idx'],  # synonym id
                'synonym': qa.get('synonym'),  # synonym name
                'template_idx': qa.get('template_idx'),  # template id
                'split': qa.get('split'),  # train / test
                # class/root-level rows (prepare_class_qa_for_sae.py): the SAE-side
                # ceiling experiment needs the node level and its descendant leaves
                'node_level': qa.get('node_level'),
                'node_idx': qa.get('node_idx'),
                'leaf_members': qa.get('leaf_members'),
                'class_idx': qa.get('class_idx'),
                'root_idx': qa.get('root_idx'),
            })

    return activations


def main():

    # ---------------------------
    # parse arguments
    # ---------------------------
    parser = argparse.ArgumentParser()

    # fine-tuned model path
    parser.add_argument('--model_path',type=str,default='models/base_lm_goal')
    # QA training data
    parser.add_argument('--train_qa',type=str,default='data/generated/qa_train_x2.jsonl')
    # output file
    parser.add_argument('--output',type=str,default='data/activations/x2m_train_activations_layer9.pkl')
    # layer to extract
    parser.add_argument('--layer',type=int,default=9,help='Layer to extract (-1 = last)')
    # maximum number of samples
    parser.add_argument('--max_samples',type=int,default=None)
    args = parser.parse_args()

    # ---------------------------
    # load model
    # ---------------------------
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    # GPT-2 has no pad token; set it manually
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------
    # device
    # ---------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    # ---------------------------
    # collect activations
    # ---------------------------
    print(f"Collecting activations from layer {args.layer}")
    activations = collect_activations(
        model,
        tokenizer,
        args.train_qa,
        layer_idx=args.layer,
        max_samples=args.max_samples
    )

    # ---------------------------
    # save activations
    # ---------------------------
    output_path = Path(args.output)

    # create directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(activations, f)
    print(f"Saved {len(activations)} activations to {output_path}")

    # print activation dimensions
    print(f"Activation shape: {activations[0]['h'].shape}")

    # ---------------------------
    # count samples per concept
    # ---------------------------
    concept_counts = {}

    for act in activations:
        concept_name = act['concept']
        concept_counts[concept_name] = concept_counts.get(concept_name, 0) + 1

    print("\nActivations per concept:")

    for concept_name, count in sorted(concept_counts.items()):
        print(f"  {concept_name}: {count}")


if __name__ == "__main__":
    main()