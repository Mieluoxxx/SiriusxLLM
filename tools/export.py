"""
此脚本包含用于模型导出的函数和工具。
我们有许多不同版本的模型，希望将它们导出为 .bin 文件，以便在 C 语言中读取并进行推理。

支持的输入版本（PyTorch 文件/模型）：
- Meta 官方发布的 Llama 2 权重
- Huggingface Hub 上可用的权重
- llama2.c（本仓库）训练的模型

支持的输出版本（.bin 文件）：
- v0：原始 llama2.c 仓库的旧版文件（最终将被弃用）
- v1-vN：改进的 .bin 文件，包含适当的头部、缓存对齐等

此脚本旨在提供所有这些转换。
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

from model import ModelArgs, Transformer


# -----------------------------------------------------------------------------
# 通用工具函数

def serialize_fp32(file, tensor):
    """将一个 fp32 张量写入以 wb 模式打开的文件"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def serialize_int8(file, tensor):
    """将一个 int8 张量写入以 wb 模式打开的文件"""
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)


def quantize_q80(w, group_size):
    """
    对张量进行 Q8_0 量化，即对称量化为 int8，范围 [-127, 127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float()  # 转换为 float32
    w = w.reshape(-1, group_size)
    # 找到每组中的最大值
    wmax = torch.abs(w).max(dim=1).values
    # 计算缩放因子，使得 float = quant * scale
    scale = wmax / 127.0
    # 缩放到范围 [-127, 127]
    quant = w / scale[:, None]
    # 四舍五入到最近的整数
    int8val = torch.round(quant).to(torch.int8)
    # 通过重新缩放反量化
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # 计算每组中的最大误差
    err = torch.abs(fp32valr - w).max(dim=1).values
    # 找到所有组中的最大误差
    maxerr = err.max().item()
    return int8val, scale, maxerr


# -----------------------------------------------------------------------------
# 旧版导出

def legacy_export(model, filepath):
    """导出 llama2.c 的旧版 bin 文件，即版本 v0"""
    out_file = open(filepath, 'wb')

    # 首先写入头部
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    # 旧版格式使用正/负词汇表大小作为共享分类器的标志
    if not shared_classifier:
        p.vocab_size = -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)

    # 接下来写入嵌入权重
    serialize_fp32(out_file, model.tok_embeddings.weight)

    # 现在写入所有层的权重
    # 注意力权重
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wq.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wk.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wv.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wo.weight)
    # FFN 权重
    for layer in model.layers:
        serialize_fp32(out_file, layer.ffn_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w1.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w2.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w3.weight)
    # 最后的 RMSNorm
    serialize_fp32(out_file, model.norm.weight)
    # freqs_cis
    serialize_fp32(out_file, model.freqs_cos[:p.max_seq_len])
    serialize_fp32(out_file, model.freqs_sin[:p.max_seq_len])

    # 最后的分类器权重
    if not shared_classifier:
        serialize_fp32(out_file, model.output.weight)

    # 写入二进制文件
    out_file.close()
    print(f"写入 {filepath}")


def legacy_export_quant(model, filepath):
    print('导出量化模型')
    """导出 llama2.c 的旧版 bin 文件，即版本 v0"""
    out_file = open(filepath, 'wb')

    # 首先写入头部
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    # 旧版格式使用正/负词汇表大小作为共享分类器的标志
    if not shared_classifier:
        p.vocab_size = -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    group_size = 64
    header = struct.pack('iiiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len, group_size)
    out_file.write(header)

    group_size = 64
    for layer in model.layers:
        q, s, err = quantize_q80(layer.attention.wq.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
    for layer in model.layers:
        q, s, err = quantize_q80(layer.attention.wk.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
    for layer in model.layers:
        q, s, err = quantize_q80(layer.attention.wv.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
    for layer in model.layers:
        q, s, err = quantize_q80(layer.attention.wo.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)

    for layer in model.layers:
        q, s, err = quantize_q80(layer.feed_forward.w1.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
    for layer in model.layers:
        q, s, err = quantize_q80(layer.feed_forward.w2.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
    for layer in model.layers:
        q, s, err = quantize_q80(layer.feed_forward.w3.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)

    # 最后的分类器权重
    if not shared_classifier:
        q, s, err = quantize_q80(model.output.weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)

    # 接下来写入嵌入权重
    serialize_fp32(out_file, model.tok_embeddings.weight)

    # 注意力权重
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention_norm.weight)

    # FFN 权重
    for layer in model.layers:
        serialize_fp32(out_file, layer.ffn_norm.weight)

    # 最后的 RMSNorm
    serialize_fp32(out_file, model.norm.weight)

    # 写入二进制文件
    out_file.close()
    print(f"写入 {filepath}")


# -----------------------------------------------------------------------------
# 新版导出

def version1_export(model, filepath):
    """
    将模型权重导出为完整的 float32 .bin 文件，以便在 C 中读取。
    与旧版导出相同，但包含适当的头部。
    """
    version = 1

    out_file = open(filepath, 'wb')
    # 首先写入头部，头部为 256 字节
    # 1) 写入魔数，即 "ak42" 的 ASCII 码
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) 写入版本号
    out_file.write(struct.pack('i', version))
    # 3) 写入参数，共 7 个整数
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) 写入其他标志
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell()  # 用零填充剩余部分
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # 现在写入所有参数
    weights = [
        *[layer.attention_norm.weight for layer in model.layers],
        *[layer.ffn_norm.weight for layer in model.layers],
        model.norm.weight,
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        serialize_fp32(out_file, w)

    # 写入二进制文件
    out_file.close()
    print(f"写入 {filepath}")


def version2_export(model, filepath, group_size=64):
    """
    将模型权重导出为 Q8_0 格式的 .bin 文件，以便在 C 中读取。
    即：
    - 将所有权重对称量化为 int8，范围 [-127, 127]
    - 其他张量（如 RMSNorm 参数）保持为 fp32
    - 量化以 group_size 为单位进行，以减少异常值的影响
    """
    version = 2

    # 首先进行一些验证
    while model.params.dim % group_size != 0:
        group_size //= 2
        print(f"回退：将 group_size 减少到 {group_size} 以适应 hidden_dim")
    weights = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        assert w.numel() % group_size == 0, f"权重 {i} 的 numel 为 {w.numel()}，不是 group_size {group_size} 的倍数"

    # 写入文件
    out_file = open(filepath, 'wb')
    # 首先写入头部，头部为 256 字节
    # 1) 写入魔数，即 "ak42" 的 ASCII 码
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) 写入版本号
    out_file.write(struct.pack('i', version))
    # 3) 写入参数，共 7 个整数
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) 写入其他标志
    out_file.write(struct.pack('B', int(shared_classifier)))
    out_file.write(struct.pack('i', group_size))  # 用于量化的 group_size
    pad = 256 - out_file.tell()  # 用零填充剩余部分
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # 首先写入所有保持为 fp32 的参数：归一化参数
    for layer in model.layers:  # 注意力归一化
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:  # MLP 归一化
        serialize_fp32(out_file, layer.ffn_norm.weight)
    serialize_fp32(out_file, model.norm.weight)  # 最后的预分类器归一化

    # 现在写入所有量化为 Q8_0 的参数
    ew = []
    for i, w in enumerate(weights):
        # 量化此权重
        q, s, err = quantize_q80(w, group_size)
        # 将 int8 权重保存到文件
        serialize_int8(out_file, q)  # 保存 int8 张量
        serialize_fp32(out_file, s)  # 保存缩放因子
        # 记录误差
        ew.append((err, w.shape))
        print(f"{i + 1}/{len(weights)} 将 {tuple(w.shape)} 量化为 Q8_0，最大误差 {err}")

    # 打印所有权重中的最大误差，应该非常小，例如 O(~0.001)
    ew.sort(reverse=True)
    print(f"所有权重中的最大量化组误差：{ew[0][0]}")

    # 写入二进制文件
    out_file.close()
    print(f"写入 {filepath}")


def hf_export(llama_model, filepath, group_size=64, dtype=torch.float32):
    """为 HuggingFace 生成 pytorch_model.bin 的 state_dict 和 config.json"""

    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except ImportError:
        print("错误：需要 transformers 包来加载 Huggingface 模型")
        print("请运行 `pip install transformers` 安装")
        return None

    # 生成 LlamaModel 的 state_dict
    hf_state_dict = {}

    # 有时我们需要为多头注意力重复键值
    dim = llama_model.params.dim
    num_key_value_heads = llama_model.params.n_kv_heads
    n_rep = llama_model.params.n_heads // num_key_value_heads
    key_value_dim = dim // n_rep

    # HuggingFace 需要权重进行置换
    def permute_original(w, n_heads=llama_model.params.n_heads, dim1=dim, dim2=dim):
        return w.view(dim1, dim2).reshape(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # 将 llama 模型的权重转移到 HF state_dict 格式
    hf_state_dict['model.embed_tokens.weight'] = llama_model.tok_embeddings.weight.clone().to(dtype)
    hf_state_dict['model.norm.weight'] = llama_model.norm.weight.clone().to(dtype)

    # 将每一层的权重添加到 HF state_dict
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

    # llama2.c 通常使用共享权重 -> 引用 embed_tokens.weights
    hf_state_dict['lm_head.weight'] = hf_state_dict['model.embed_tokens.weight']

    # 检查嵌入是否共享，否则使用手动输出权重
    _embeddings_are_tied: bool = torch.equal(llama_model.tok_embeddings.weight, llama_model.output.weight)
    if not _embeddings_are_tied:
        hf_state_dict['lm_head.weight'] = llama_model.output.weight.clone().to(dtype)

    # 生成 LlamaConfig

    # 从 llama.c 模型中提取必要的属性
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
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
    )

    # 在 filepath 目录中保存文件
    os.makedirs(filepath, exist_ok=True)

    # 将 state_dict 保存为 .bin 格式，config 保存为 .json
    torch.save(hf_state_dict, os.path.join(filepath, "pytorch_model.bin"))
    config.save_pretrained(filepath)


# -----------------------------------------------------------------------------
# 加载/导入函数

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


def load_meta_model(model_path):
    params_path = os.path.join(model_path, 'params.json')
    with open(params_path) as f:
        params = json.load(f)
        print(params)

    model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))
    models = [torch.load(p, map_location='cpu') for p in model_paths]

    def concat_weights(models):
        state_dict = {}
        for name in list(models[0]):
            tensors = [model[name] for model in models]
            if len(tensors) == 1 or len(tensors[0].shape) == 1:
                state_dict[name] = tensors[0]
                continue
            is_axis_1 = (
                    name.startswith('tok_embeddings.')
                    or name.endswith('.attention.wo.weight')
                    or name.endswith('.feed_forward.w2.weight')
            )
            axis = 1 if is_axis_1 else 0
            state_dict[name] = torch.cat(tensors, dim=axis)
            for model in models:
                del model[name]
        return state_dict

    state_dict = concat_weights(models)
    del models

    # 设置 ModelArgs
    config = ModelArgs()
    config.dim = params["dim"]
    config.n_layers = params["n_layers"]
    config.n_heads = params["n_heads"]
    config.n_kv_heads = params.get('n_kv_heads') or params['n_heads']
    config.multiple_of = params["multiple_of"]
    config.norm_eps = params["norm_eps"]

    config.vocab_size = state_dict['tok_embeddings.weight'].shape[0]
    config.max_seq_len = 2048

    # 创建一个新的 Transformer 对象并设置权重
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(state_dict['tok_embeddings.weight'])
    model.norm.weight = nn.Parameter(state_dict['norm.weight'])

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(state_dict[f'layers.{i}.attention_norm.weight'])
        layer.attention.wq.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wq.weight'])
        layer.attention.wk.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wk.weight'])
        layer.attention.wv.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wv.weight'])
        layer.attention.wo.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wo.weight'])
        layer.ffn_norm.weight = nn.Parameter(state_dict[f'layers.{i}.ffn_norm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w1.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w2.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w3.weight'])

    # 最后的分类器
    model.output.weight = nn.Parameter(state_dict['output.weight'])
    model.eval()
    return model


def load_hf_model(model_path):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("错误：需要 transformers 包来加载 Huggingface 模型")
        print("请运行 `pip install transformers` 安装")
        return None

    # 加载 HF 模型
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_dict = hf_model.state_dict()

    # 将 LlamaConfig 转换为 ModelArgs
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

    # 创建一个新的 Transformer 对象并设置权重
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(hf_dict['model.embed_tokens.weight'])
    model.norm.weight = nn.Parameter(hf_dict['model.norm.weight'])

    # Huggingface 对 WQ 和 WK 进行了置换，此函数将其反转
    def permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

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

    # 最后的分类器
    model.output.weight = nn.Parameter(hf_dict['lm_head.weight'])
    model.eval()
    return model


# -----------------------------------------------------------------------------
# API 入口点

def model_export(model, filepath, version, dtype=torch.float32):
    """
    版本说明：
    v-1：Huggingface 导出，即用于此仓库之外，在 HF 中使用
    v0：旧版 llama2.c float 格式，已弃用
    v1：float32 导出
    v2：int8 量化的 Q8_0 导出，类似于 llama.cpp，分组量化
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
        hf_export(model, filepath, dtype)
    else:
        raise ValueError(f"未知版本 {version}")


# -----------------------------------------------------------------------------
# CLI 入口点

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="输出文件路径")
    parser.add_argument("--version", default=0, type=int, help="导出版本")
    parser.add_argument("--dtype", type=str, help="模型的数据类型（fp16, fp32）", default="fp32")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="模型检查点，.pt 文件")
    group.add_argument("--meta-llama", type=str, help="Meta Llama 模型路径")
    group.add_argument("--hf", type=str, help="Huggingface 模型路径")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)
    elif args.meta_llama:
        model = load_meta_model(args.meta_llama)
    elif args.hf:
        model = load_hf_model(args.hf)

    if model is None:
        parser.error("无法加载输入模型！")

    # 导出
    model_export(model, args.filepath, args.version, args.dtype)