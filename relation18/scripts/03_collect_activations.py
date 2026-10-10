"""
Step 3: Collect layer-9 last-token activations from the frozen SFT model.

Step 5: Collect Activations from SFT Model
Extract hidden states at the answer position for SAE training.
"""


import os  # 环境变量（CUDA 设备顺序等）
import json  # 读取json/jsonl数据
import torch  # PyTorch
import numpy as np  # 数值计算
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace模型
from pathlib import Path  # 路径处理
import argparse  # 命令行参数解析
from tqdm import tqdm  # 进度条
import pickle  # 用于保存python对象


import _bootstrap


def collect_activations(model, tokenizer, qa_file, layer_idx=-1, max_samples=None):
    """
    从模型指定层收集激活向量
    
    使用纯QA格式（不带biography上下文）
    这样模型必须依赖内部知识回答
    
    参数：
        model: 已训练好的语言模型
        tokenizer: 对应tokenizer
        qa_file: QA数据文件
        layer_idx: 要提取的层 (-1表示最后一层)
        max_samples: 最大样本数量
    """

    # 获取模型所在设备（GPU / CPU）
    device = next(model.parameters()).device

    # ---------------------------
    # 读取QA数据
    # ---------------------------
    with open(qa_file, 'r') as f:
        qa_pairs = [json.loads(line) for line in f]

    # 如果指定最大样本数量，则截断
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]

    # 用于存储最终激活
    activations = []

    # 设置模型为eval模式
    model.eval()

    # 不需要梯度计算（节省显存）
    with torch.no_grad():

        # 遍历所有QA样本
        for qa in tqdm(qa_pairs, desc="Collecting activations"):

            # ---------------------------
            # 构造输入prompt
            # ---------------------------
            # 这里只给问题，不给biography
            # 目的是测试模型是否记住知识
            prompt = f"Q: {qa['question']}\nA:"

            # tokenizer编码
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            # ---------------------------
            # 前向传播
            # ---------------------------
            # output_hidden_states=True 会返回所有层的hidden states
            outputs = model(**inputs, output_hidden_states=True)

            # ---------------------------
            # 提取指定层的hidden states
            # ---------------------------
            # hidden_states 是一个tuple：
            # (embedding, layer1, layer2, ... layerN)

            hidden_states = outputs.hidden_states[layer_idx]  # shape: [1, seq_len, d_model]

            # 获取prompt最后一个token的位置
            # 也就是模型准备生成答案的位置
            last_position = inputs['input_ids'].shape[1] - 1

            # 提取该位置的hidden state
            # shape: [d_model]
            h = hidden_states[0, last_position, :].cpu().numpy()

            # ---------------------------
            # 处理答案token
            # ---------------------------
            # 将答案tokenize
            answer_tokens = tokenizer.encode(
                f" {qa['answer']}",
                add_special_tokens=False
            )

            # ---------------------------
            # 保存信息
            # ---------------------------
            activations.append({
                'h': h,  # hidden state向量
                'concept_idx': qa['concept_idx'],  #  关系ID like 0
                'concept': qa.get('concept'),  # 关系名称 like "dog"
                'answer': qa['answer'],  # 答案文本
                'answer_tokens': answer_tokens,  # 答案token
                'property_idx': qa['property_idx'],  # property ID
                'property': qa.get('property'),  # property 名称
                'question': qa['question'],  # 问题文本
                'synonym_idx': qa['synonym_idx'],  # 同义词ID
                'synonym': qa.get('synonym'),  # 同义词名称
                'template_idx': qa.get('template_idx'),  # 模板ID
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
    # 解析命令行参数
    # ---------------------------
    parser = argparse.ArgumentParser()

    # SFT模型路径
    parser.add_argument('--model_path',type=str,default='models/base_lm_goal')
    # QA训练数据
    parser.add_argument('--train_qa',type=str,default='data/generated/qa_train.jsonl')
    # 输出文件
    parser.add_argument('--output',type=str,default='data/activations/goal_train_layer9.pkl')
    # 要提取的layer
    parser.add_argument('--layer',type=int,default=9,help='Layer to extract (-1 = last)')
    # 最大样本数量
    parser.add_argument('--max_samples',type=int,default=None)
    args = parser.parse_args()

    # ---------------------------
    # 加载模型
    # ---------------------------
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    # GPT2没有pad token，需要手动设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------
    # 设备配置
    # ---------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    # ---------------------------
    # 收集激活
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
    # 保存激活
    # ---------------------------
    output_path = Path(args.output)

    # 创建目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为pickle
    with open(output_path, 'wb') as f:
        pickle.dump(activations, f)
    print(f"Saved {len(activations)} activations to {output_path}")

    # 打印激活向量维度
    print(f"Activation shape: {activations[0]['h'].shape}")

    # ---------------------------
    # 统计每个concept的样本数量
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