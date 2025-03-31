"""
This script has functions and utilties for model export.
Basically, we have a bunch of versions of the model, and we
want to export them to .bin files to be read from and inferenced in C.

Among the "input" versions of PyTorch files/models:
- Official Llama 2 weights released by Meta
- Huggingface weights available on the hub
- llama2.c (this repo) trained models

Among the "output" versions of .bin files:
- v0: Legacy files of the original llama2.c repo (will eventually be DEPRECATED)
- v1-vN: Improved .bin files with a proper header, cache alignment, etc.

This script aspires to provide all of these conversions.
"""
import os
import gzip
import shutil
import struct
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

# 导入通用导出功能
from export import (
    serialize_fp32, serialize_int8, quantize_q80,
    legacy_export as base_legacy_export,
    legacy_export_quant as base_legacy_export_quant,
    version1_export as base_version1_export,
    version2_export as base_version2_export,
    model_export as base_model_export
)

from model_qwen2 import ModelArgs, Transformer


# 加载模型的函数
def load_checkpoint(checkpoint):
    # 加载提供的模型检查点
    checkpoint_dict = torch.load(checkpoint, map_location='cpu')
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# 加载HF模型的函数
def load_hf_model(model_path):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # 加载HF模型
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_dict = hf_model.state_dict()

    # 转换LlamaConfig为ModelArgs
    config = ModelArgs()
    if any(['config.json' in path for path in os.listdir("./")]):
        with open(os.path.join("./", 'config.json'), 'r') as f:
            config_json = json.load(f)
        config.dim = config_json["hidden_size"]
        config.n_layers = config_json["num_hidden_layers"]
        config.n_heads = config_json["num_attention_heads"]
        config.n_kv_heads = config_json["num_key_value_heads"]
        config.vocab_size = config_json["vocab_size"]
        config.hidden_dim = config_json["intermediate_size"]
        config.norm_eps = config_json["rms_norm_eps"]
        config.max_seq_len = config_json["max_position_embeddings"]
    else:
        config.dim = hf_model.config.hidden_size
        config.n_layers = hf_model.config.num_hidden_layers
        config.n_heads = hf_model.config.num_attention_heads
        config.n_kv_heads = hf_model.config.num_key_value_heads
        config.vocab_size = hf_model.config.vocab_size
        config.hidden_dim = hf_model.config.intermediate_size
        config.norm_eps = hf_model.config.rms_norm_eps
        config.max_seq_len = hf_model.config.max_position_embeddings

    # 创建新的Transformer对象并设置权重
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(hf_dict['model.embed_tokens.weight'])
    model.norm.weight = nn.Parameter(hf_dict['model.norm.weight'])

    # 设置每一层的权重
    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.input_layernorm.weight'])
        layer.attention.wq.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.q_proj.weight'])
        layer.attention.wk.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.k_proj.weight'])
        layer.attention.wv.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.v_proj.weight'])
        layer.attention.wo.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.o_proj.weight'])
        layer.ffn_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.post_attention_layernorm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.gate_proj.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.down_proj.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.up_proj.weight'])
        
        # 处理偏置（如果存在）
        if hasattr(layer.attention.wq, 'bias') and f'model.layers.{i}.self_attn.q_proj.bias' in hf_dict:
            layer.attention.wq.bias = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.q_proj.bias'])
        if hasattr(layer.attention.wk, 'bias') and f'model.layers.{i}.self_attn.k_proj.bias' in hf_dict:
            layer.attention.wk.bias = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.k_proj.bias'])
        if hasattr(layer.attention.wv, 'bias') and f'model.layers.{i}.self_attn.v_proj.bias' in hf_dict:
            layer.attention.wv.bias = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.v_proj.bias'])

    # 最终分类器
    model.output.weight = nn.Parameter(hf_dict['lm_head.weight'])
    model.eval()
    return model

# 导出到HF格式
def hf_export(llama_model, filepath, group_size=64, dtype=torch.float32):
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # 生成LlamaModel状态字典
    hf_state_dict = {}

    # 有时我们对头部有重复的键值
    dim = llama_model.params.dim
    num_key_value_heads = llama_model.params.n_kv_heads
    n_rep = llama_model.params.n_heads // num_key_value_heads
    key_value_dim = dim // n_rep

    # HuggingFace需要权重进行排列
    def permute_original(w, n_heads=llama_model.params.n_heads, dim1=dim, dim2=dim):
        return w.view(dim1, dim2).reshape(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # 将权重从llama模型转移到HF状态字典格式
    hf_state_dict['model.embed_tokens.weight'] = llama_model.tok_embeddings.weight.clone().to(dtype)
    hf_state_dict['model.norm.weight'] = llama_model.norm.weight.clone().to(dtype)

    # 添加每一层的权重到HF状态字典
    for i, layer in enumerate(llama_model.layers):
        layer_id = layer.layer_id
        hf_state_dict[f'model.layers.{i}.input_layernorm.weight'] = llama_model.layers[
            layer_id].attention_norm.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = permute_original(
            llama_model.layers[layer_id].attention.wq.weight.clone()).to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = permute_original(
            llama_model.layers[layer_id].attention.wk.weight.clone(), num_key_value_heads, key_value_dim, dim).to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = llama_model.layers[
            layer_id].attention.wv.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = llama_model.layers[
            layer_id].attention.wo.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = llama_model.layers[
            layer_id].ffn_norm.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = llama_model.layers[
            layer_id].feed_forward.w1.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.down_proj.weight'] = llama_model.layers[
            layer_id].feed_forward.w2.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.up_proj.weight'] = llama_model.layers[
            layer_id].feed_forward.w3.weight.clone().to(dtype)
            
        # 处理偏置（如果存在）
        if hasattr(llama_model.layers[layer_id].attention.wq, 'bias'):
            hf_state_dict[f'model.layers.{i}.self_attn.q_proj.bias'] = llama_model.layers[
                layer_id].attention.wq.bias.clone().to(dtype)
        if hasattr(llama_model.layers[layer_id].attention.wk, 'bias'):
            hf_state_dict[f'model.layers.{i}.self_attn.k_proj.bias'] = llama_model.layers[
                layer_id].attention.wk.bias.clone().to(dtype)
        if hasattr(llama_model.layers[layer_id].attention.wv, 'bias'):
            hf_state_dict[f'model.layers.{i}.self_attn.v_proj.bias'] = llama_model.layers[
                layer_id].attention.wv.bias.clone().to(dtype)

    # llama2.c通常使用绑定权重 -> 引用embed_tokens.weights
    hf_state_dict['lm_head.weight'] = hf_state_dict['model.embed_tokens.weight']

    # 检查嵌入是否绑定，否则使用手动输出权重
    _embeddings_are_tied: bool = torch.equal(llama_model.tok_embeddings.weight, llama_model.output.weight)
    if not _embeddings_are_tied:
        hf_state_dict['lm_head.weight'] = llama_model.output.weight.clone().to(dtype)

    # 生成LlamaConfig

    # 从llama.c模型中提取必要的属性
    vocab_size = llama_model.params.vocab_size
    hidden_size = llama_model.params.dim
    intermediate_size = llama_model.layers[0].feed_forward.w1.weight.shape[0]
    num_hidden_layers = llama_model.params.n_layers
    num_attention_heads = llama_model.params.n_heads
    num_key_value_heads = llama_model.params.n_kv_heads
    max_position_embeddings = llama_model.params.max_seq_len
    rms_norm_eps = llama_model.params.norm_eps

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        tie_word_embeddings=_embeddings_are_tied,
        # 手动设置
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
    )

    # 在目录filepath中保存文件
    # 如果目录不存在，先创建
    os.makedirs(filepath, exist_ok=True)

    # 将状态字典保存为.bin格式，配置保存为.json
    torch.save(hf_state_dict, os.path.join(filepath, "pytorch_model.bin"))
    config.save_pretrained(filepath)

# 重写导出函数，使用Qwen2特定的处理
def legacy_export(model, filepath):
    base_legacy_export(model, filepath)

def legacy_export_quant(model, filepath):
    base_legacy_export_quant(model, filepath)

def version1_export(model, filepath):
    base_version1_export(model, filepath)

def version2_export(model, filepath):
    base_version2_export(model, filepath)

def model_export(model, filepath, version, dtype=torch.float32):
    """
    版本说明:
    v-1: huggingface导出，即用于此仓库外部的HF
    v0: 旧版llama2.c浮点格式，已弃用
    v1: float32导出
    v2: int8量化Q8_0导出，类似于llama.cpp，分组
    """
    if version == 0:
        legacy_export(model, filepath)
    elif version == 1:
        version1_export(model, filepath)
    elif version == 2:
        version2_export(model, filepath)
    elif version == 3:
        legacy_export_quant(model, filepath)
    elif version == -1:
        hf_export(model, filepath, dtype=dtype)
    else:
        raise ValueError(f"未知版本 {version}")

# 主函数和命令行参数处理可以保留原样


def torchscript_export(model, filepath, zero_params=False, gzip_output=False):
    """
    (This was submitted via a PR earlier. Leaving it here, but "orphaned" for now)
    Saves the model as a TorchScript.
    The resulting file can be loaded in C++ code and then used for training or
    inference with:
        #include <torch/script.h>
        torch::jit::Module module = torch::jit::load("model.pt")
    Note that the serialized model includes the initial parameters and with the default
    ModelArgs the file is 59M and gzips down to 55M. If you want to serialize/distribute
    the model parameters separately you can zero out the parameters before saving it and
    it will gzip down to 780K.
    """

    # If requested zero params before saving the model. This is useful in
    # conjunction with gzip_output.
    if zero_params:
        for p in model.parameters():
            p.detach().zero_()

    torch.jit.save(torch.jit.script(model), filepath)

    if gzip_output:
        with open(filepath, "rb") as f_in:
            with gzip.open(f"{filepath}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.unlink(filepath)


# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--version", default=0, type=int, help="the version to export with")
    parser.add_argument("--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    group.add_argument("--meta-llama", type=str, help="meta llama model path")
    group.add_argument("--hf", type=str, help="huggingface model path")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)
    elif args.meta_llama:
        model = load_meta_model(args.meta_llama)
    elif args.hf:
        model = load_hf_model(args.hf)

    if model is None:
        parser.error("Can't load input model!")

    # export
    model_export(model, args.filepath, args.version, args.dtype)