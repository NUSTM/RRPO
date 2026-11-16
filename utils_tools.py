import torch
import torch.nn as nn

import json
import pickle
import os
import sys
import math
from copy import deepcopy
from datetime import datetime
import pytz


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_model_grads(model: nn.Module):
    """打印每一层的梯度"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Layer: {name}, Grad Norm: {param.grad.norm().item()}")
        else:
            print(f"Layer: {name}, No gradient")


def freeze_model(model: nn.Module):
    """冻结模型所有参数"""
    for param in model.parameters():
        param.requires_grad = False


def get_search_doc(index: str, ids: list[str]) -> list[dict]:
    """构造 Elasticsearch mget 请求文档。
    index: 索引名
    ids: 文档 id 列表
    """
    return [{"_index": index, "_id": id} for id in ids]


def load_results(load_path: str):
    """加载 json 或 pkl 文件"""
    if load_path.endswith(".json"):
        with open(load_path, "r") as f:
            results = json.load(f)
    elif load_path.endswith(".pkl"):
        with open(load_path, "rb") as f:
            results = pickle.load(f)
    else:
        raise ValueError("load_path must be a json or pkl file")

    return results


def save_results(results, save_path: str):
    """保存为 json 或 pkl 文件"""
    if save_path.endswith(".json"):
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
    elif save_path.endswith(".pkl"):
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
    else:
        raise ValueError("save_path must be a json or pkl file")


def print_time(return_str_instead: bool = False):
    """打印当前 UTC+8 时间。依赖 datetime 与 pytz。"""
    utc_tz = pytz.timezone('UTC')
    utc8_tz = pytz.timezone('Asia/Shanghai')

    utc_now = datetime.now(utc_tz)
    utc8_now = utc_now.astimezone(utc8_tz)

    message_str = "Current Time: (UTC+8):" + utc8_now.strftime('%Y-%m-%d %H:%M:%S')
    if return_str_instead:
        return message_str
    else:
        print(message_str)


def compare_models(model_a: nn.Module, model_b: nn.Module, eps: float = 1e-8):
    """比较两个模型的参数是否一致"""
    for (name_a, param_a), (name_b, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        if name_a != name_b:
            print(f"参数名称不一致: {name_a} vs {name_b}")
            return False
        if not torch.allclose(param_a.data, param_b.data, atol=eps):
            print(f"参数不一致: {name_a}")
            print(param_a.data)
            print(param_b.data)
            return False
    print("所有参数在两个模型中都是一致的。")
    return True


def redirect_output(rank, log_dir, log_filename):
    """将 stdout 和 stderr 重定向到同一个文件。

    Args:
        rank (int): 当前进程的 rank。
        log_dir (str): 日志目录。
        log_filename (str): 日志文件名，不要加后缀。
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{log_filename}-rank{rank}.log')
    f = open(log_file, 'w')
    sys.stdout = f
    sys.stderr = f


def merge_dicts(*dicts):
    """合并多个字典。键冲突且值不同时保留新值，并打印冲突信息。

    使用深拷贝，避免修改原始字典。建议值为 Python 基本类型或其组合。
    用最长字典的深拷贝初始化结果，以减少后续拷贝。
    """
    all_lens = [len(d) for d in dicts]
    max_len_idx = all_lens.index(max(all_lens))
    merged_dict = deepcopy(dicts[max_len_idx])

    for d in dicts:
        for key, value in d.items():
            if key in merged_dict:
                existing_value = merged_dict[key]
                if existing_value != value:
                    print(f"冲突的键 '{key}'，值分别为 '{existing_value}' 和 '{value}'. 合并的字典保留新值")
                    merged_dict[key] = deepcopy(value)
            merged_dict[key] = deepcopy(value)
    return merged_dict


def compute_logits(model: nn.Module, model_inputs: dict[str, torch.Tensor]):
    """计算 logits。torch.no_grad() 或 torch.autocast() 等上下文管理器由调用方负责。"""
    out = model(**model_inputs, return_dict=True)

    logits = out.logits.view(-1, )
    return logits


def sort_rerank_result_and_get_action_passages(rerank_result: dict[str, any], top_k: int) -> list[str]:
    """按 scores 降序排列，并取出 top-k passages。
    同时重排 scores、passages，以及存在时的 passage_ids。
    """
    if 'passage_ids' in rerank_result:
        passage_ids = rerank_result['passage_ids']
    scores = rerank_result['scores']
    passages = rerank_result['passages']

    assert len(scores) == len(passages), "Passages和Scores的长度不一致"

    if isinstance(scores, list):
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    elif isinstance(scores, torch.Tensor):
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_indices = sorted_indices.tolist()

    sorted_rerank_result = deepcopy(rerank_result)
    if 'passage_ids' in rerank_result:
        sorted_rerank_result['passage_ids'] = [passage_ids[i] for i in sorted_indices]
    if isinstance(scores, list):
        sorted_rerank_result['scores'] = [scores[i] for i in sorted_indices]
    elif isinstance(scores, torch.Tensor):
        sorted_rerank_result['scores'] = sorted_scores
    sorted_rerank_result['passages'] = [passages[i] for i in sorted_indices]

    action_passages = [passages[sorted_indices[i]] for i in range(top_k)]

    return sorted_rerank_result, action_passages


def compute_single_case_logits(model: nn.Module, model_inputs: dict[str, torch.Tensor], chunk_size: int, model_device: torch.device = None):
    """按 chunk 计算 logits。混合精度和 torch.no_grad() 由调用方负责。
    batch_len 不必被 chunk_size 整除。
    """
    batch_len = model_inputs['input_ids'].shape[0]

    all_logits = []
    forward_steps_count = math.ceil(batch_len / chunk_size)
    for forward_step in range(forward_steps_count):
        start_chunk_idx = forward_step * chunk_size
        end_chunk_idx = min((forward_step + 1) * chunk_size, batch_len)

        if model_device is None:
            model_inputs_chunk = {k: v[start_chunk_idx:end_chunk_idx] for k, v in model_inputs.items()}
        else:
            model_inputs_chunk = {k: v[start_chunk_idx:end_chunk_idx].to(model_device) for k, v in model_inputs.items()}

        if model_inputs_chunk['input_ids'].shape[0] == 1:
            for k, v in model_inputs_chunk.items():
                model_inputs_chunk[k] = v.reshape(1, -1)

        chunk_logits = compute_logits(model, model_inputs_chunk)
        all_logits.append(chunk_logits)

    all_logits = torch.cat(all_logits, dim=0)

    return all_logits


def mycollate_fn(batch: list):
    """仅供 batch_size=1 使用，避免默认 collate_fn 把数据包装成 tensor。"""
    return batch[0]


class CollatorWithTokenizer:
    """使用 tokenizer 处理数据的 collate_fn。
    建议 batch_size=1 以缩短序列长度。代码兼容更大的 batch_size。
    """
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """将 batch 中的 query 与 passages 组成 pair 后编码。"""
        queries = [item["query"] for item in batch]
        passages = [item["passages"] for item in batch]

        pairs = [[q, p] for q, ps in zip(queries, passages) for p in ps]

        input_tensors = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        batch.update(input_tensors)
        return batch


def post_process_on_base_model_response(response: str):
    """截断模型响应：遇到换行即截止，并去掉开头的 "1. " 编号。"""
    catch_patterns = ["\n"]

    for pattern in catch_patterns:
        if pattern in response:
            response = response.split(pattern)[0]
            break

    if response.startswith("1. "):
        if " 2. " in response:
            idx = response.find(' 2.')
            response = response[:idx].strip()

        response = response[2:]

    response = response.strip()

    return response
