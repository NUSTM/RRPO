# -*- coding: utf-8 -*-
import torch
import random
from heapq import nlargest
import math
from collections import deque
import pickle
import requests
from transformers import PreTrainedTokenizerBase

from typing import Union

from utils_tools import *
from utils_metrics import *


def get_softmax_probs_from_logits_with_mask(logits: torch.Tensor, mask: torch.Tensor, large_negative_number: int) -> torch.Tensor:
    """将 mask 为 0 的位置换成大负数后计算 softmax。用 logsumexp 避免下溢。"""
    logits_masked = torch.where(mask == 0, large_negative_number, logits)
    softmax_probs = torch.exp(logits_masked - (torch.logsumexp(logits_masked, dim=0)))
    return softmax_probs


def select_action_from_distribution(
        logits: torch.Tensor,
        k: int,
        temperature: float = 1.0,
        return_probs: bool = False,
        device: torch.device = None
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """从分布中无放回采样 k 个动作，并返回对数概率。temperature 控制 softmax 温度。"""
    new_tensor_device = torch.device('cuda') if device is None else device
    logits_dtype = logits.dtype

    large_negative_number = get_large_negative_number(logits)

    action = []
    action_probs = []
    action_log_prob = torch.zeros(1, dtype=logits_dtype, device=new_tensor_device)
    logits_scaled = logits / temperature
    for i in range(k):
        # 每一步重建 mask。在循环外原地修改会导致梯度无法传播。
        mask = torch.ones(logits.size(0), device=new_tensor_device)
        for j in range(i):
            mask[action[j]] = 0

        softmax_probs = get_softmax_probs_from_logits_with_mask(logits_scaled, mask, large_negative_number)

        # 再乘 mask，把已选位置置 0。Categorical 会对 probs 再做归一化。
        category_probs = softmax_probs * mask

        dist = torch.distributions.Categorical(category_probs)
        sample = dist.sample()

        action.append(sample)
        action_probs.append(softmax_probs[action[i]])
        action_log_prob += torch.log(softmax_probs[action[i]])

    action = torch.stack(action)
    if return_probs:
        action_probs = torch.stack(action_probs)
        return action, action_log_prob, action_probs
    else:
        return action, action_log_prob


def select_max_prob_action(
        logits: torch.Tensor,
        top_k: int,
        temperature: float = 1.0,
        return_probs: bool = False,
        device: torch.device = None
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """取概率最大的 top-k 动作及其对数概率。temperature 控制 softmax 温度。"""
    new_tensor_device = torch.device('cuda') if device is None else device
    logits_dtype = logits.dtype

    large_negative_number = get_large_negative_number(logits)

    sorted_logits, original_indices = torch.sort(logits, descending=True)
    action = original_indices[:top_k]

    action_probs = []
    action_log_prob = torch.zeros(1, dtype=logits_dtype, device=new_tensor_device)
    logits_scaled = logits / temperature
    for i in range(top_k):
        # 每一步重建 mask。在循环外原地修改会导致梯度无法传播。
        mask = torch.ones(logits.size(0), device=new_tensor_device)
        for j in range(i):
            mask[action[j]] = 0

        softmax_probs = get_softmax_probs_from_logits_with_mask(logits_scaled, mask, large_negative_number)

        action_probs.append(softmax_probs[action[i]])
        action_log_prob += torch.log(softmax_probs[action[i]])

    if return_probs:
        action_probs = torch.stack(action_probs)
        return action, action_log_prob, action_probs
    else:
        return action, action_log_prob


def get_action_log_prob(
        logits: torch.Tensor,
        action: torch.Tensor,
        temperature: float = 1.0,
        return_probs: bool = False,
        device: torch.device = None
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """计算给定动作的对数概率，主要供 PPO 训练使用。temperature 控制 softmax 温度。"""
    new_tensor_device = torch.device('cuda') if device is None else device
    logits_dtype = logits.dtype

    large_negative_number = get_large_negative_number(logits)

    action_probs = []
    action_log_prob = torch.zeros(1, dtype=logits_dtype, device=new_tensor_device)
    logits_scaled = logits / temperature
    for i in range(action.size(0)):
        # 每一步重建 mask。在循环外原地修改会导致梯度无法传播。
        mask = torch.ones(logits.size(0), device=new_tensor_device)
        for j in range(i):
            mask[action[j]] = 0

        softmax_probs = get_softmax_probs_from_logits_with_mask(logits_scaled, mask, large_negative_number)

        action_probs.append(softmax_probs[action[i]])
        action_log_prob += torch.log(softmax_probs[action[i]])

    if return_probs:
        action_probs = torch.stack(action_probs)
        return action_log_prob, action_probs
    else:
        return action_log_prob


def compute_GAE(rewards: Union[list[int], torch.Tensor], v_values: Union[list[int], torch.Tensor], device: torch.device = None, gamma=0.99, lam=0.95, dtype = None) -> torch.Tensor:
    """计算广义优势估计。返回前切断梯度。"""
    deltas = torch.zeros(len(rewards), device=torch.device('cuda') if device is None else device, dtype=dtype if dtype else torch.float32)
    for t in range(deltas.shape[0] - 1):
        deltas[t] = rewards[t] + gamma * v_values[t + 1] - v_values[t]
    deltas[-1] = rewards[-1] - v_values[-1]

    advantages = torch.zeros_like(deltas, dtype=deltas.dtype)
    gae = 0
    for t in reversed(range(len(deltas))):
        gae = deltas[t] + gamma * lam * gae
        advantages[t] = gae

    return advantages.detach()


def add_noises_to_action(numbers: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
    """随机替换 choice 中的若干位置，替换值来自 numbers 里未被选中的数。"""
    # 1. 从 choice 中随机选取 k 个数 (k是从 0 到 len(choice) 随机选)
    k = random.randint(0, len(choice))

    # 2. 从 numbers - choice 中选取 k 个不重复的数
    remaining_numbers = set(numbers.tolist()) - set(choice.tolist())
    shifts = random.sample(remaining_numbers, k)

    # 3. 用 shifts 中的数替换 "需要改变的数"
    choice_list = choice.tolist()

    indices_to_replace = random.sample(range(len(choice_list)), k)

    for i, idx in enumerate(indices_to_replace):
        choice_list[idx] = shifts[i]

    return torch.tensor(choice_list)


def get_unique_and_indices(a: list[str]) -> tuple[list[str], list[int]]:
    """去重并保持首次出现的顺序，返回唯一值列表及每个原元素对应的下标。"""
    unique_map = {}
    b = []
    for idx, item in enumerate(a):
        if item not in unique_map:
            unique_map[item] = len(b)
            b.append(item)

    indices = [unique_map[item] for item in a]

    return b, indices


def get_top_k_from_results_to_res(data, k) -> list:
    """取每个 query 分数最高的前 k 条，结果仍按分数降序。

    :param data: dict{query_id: dict{passage_id: passage_score}}
    :param k: 保留的条数
    :return: [[query_id, [[passage_id, passage_score], ...]], ...]
    """
    res = []
    for query_id, passages in data.items():
        top_k = nlargest(k, passages.items(), key=lambda x: x[1])
        res.append([query_id, [[passage_id, score] for passage_id, score in top_k]])
    return res


def get_top_k_from_res_to_results(data, k) -> dict:
    """把嵌套列表转回按 query 分组的字典，并只保留分数最高的前 k 条。

    :param data: [[query_id, [[passage_id, passage_score], ...]], ...]
    :param k: 保留的条数
    :return: dict{query_id: dict{passage_id: passage_score}}
    """
    results = {}
    for query_entry in data:
        query_id, passages = query_entry
        top_k = nlargest(k, passages, key=lambda x: x[1])
        results[query_id] = {passage_id: score for passage_id, score in top_k}
    return results


class SeedGenerator:
    def __init__(self):
        self.seed = 0
        self.start_seed = 0

    def generate_seed(self) -> int:
        self.seed += 1
        return self.seed

    def get_all_seeds(self) -> list[int]:
        """返回已生成过的种子。"""
        return list(range(self.start_seed + 1, self.seed + 1))

    def set_seed(self, seed) -> None:
        """重新设置随机数种子"""
        self.seed = seed
        self.start_seed = seed


def get_large_negative_number(logits: torch.Tensor) -> int:
    """按 dtype 选择 mask 用的大负数。float16 用 -6e4，bfloat16 和 float32 用 -1e9。"""
    logits_dtype = logits.dtype

    if logits.is_floating_point():
        if logits_dtype == torch.bfloat16 or logits_dtype == torch.float32:
            large_negative_number = -1e9
        elif logits_dtype == torch.float16:
            large_negative_number = -6e4
        else:
            large_negative_number = -1e9
    else:
        raise TypeError(f"Unsupported tensor dtype: {logits_dtype}. The tensor must be of a floating point type.")

    return large_negative_number


def compute_kl_divergence_with_mask(p: torch.Tensor, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """计算 KL(p || q)。mask 为 0 的位置不参与计算，并避开 log(0)。"""
    p_refined = p * mask + (1 - mask) * torch.e
    q_refined = q * mask + (1 - mask) * torch.e

    p_log = p_refined.log()
    q_log = q_refined.log()

    p_log_masked = p_log * mask
    q_log_masked = q_log * mask

    p_masked_zero = p * mask + (1 - mask) * 0

    kl_div = torch.sum(p_masked_zero * (p_log_masked - q_log_masked), dim=-1)

    return kl_div


def get_probs_from_logits_with_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """由 logits 得到概率。mask 为 0 的位置换成大负数，不参与 softmax。"""
    large_negative_number = get_large_negative_number(logits)
    logits_masked = logits * mask + (1 - mask) * large_negative_number
    softmax_probs = torch.exp(logits_masked - (torch.logsumexp(logits_masked, dim=0)))
    return softmax_probs


def get_episode_v_responses_from_cache(reader_results_bm25_reranker_raw: dict, case_id: str, train_top_k: int) -> list[str]:
    """读取未训练 reranker 在各时间步缓存的回答。"""
    reranker_per_action_responses = [reader_results_bm25_reranker_raw[case_id][str(i)]["response"] for i in range(1, train_top_k + 1)]
    return reranker_per_action_responses


def compute_refined_rewards(
    rewards: Union[torch.Tensor | list],
    p: torch.Tensor,
    q: torch.Tensor,
    action: Union[torch.Tensor | list],
    device: torch.device = None,
    return_kl_divs: bool = False
    ) -> torch.Tensor:
    """用 KL 散度调整奖励：只从最后一个时间步减去 KL，其余时间步交给 GAE。

    Args:
        rewards (torch.Tensor | list): 原始奖励，shape 为 [T]。
        p (torch.Tensor): 参考分布的 logits，shape 为 [to_rerank_passages_num]。
        q (torch.Tensor): 当前分布的 logits，shape 为 [to_rerank_passages_num]。
        action (torch.Tensor | list): 选择的动作，shape 为 [T]。
        device (torch.device): tensor 所在设备。
        return_kl_divs (bool): 是否同时返回 KL。

    Returns:
        refined_rewards (torch.Tensor): 调整后的奖励。
    """
    if isinstance(rewards, list):
        rewards = torch.tensor(rewards, dtype=p.dtype, device=torch.device('cuda') if device is None else device)

    # 对完整分布计算 KL，mask 全为 1。
    mask = torch.ones(p.size(0), device=torch.device('cuda') if device is None else device)
    p_t = get_probs_from_logits_with_mask(p, mask)
    q_t = get_probs_from_logits_with_mask(q, mask)

    kl_div = compute_kl_divergence_with_mask(p_t, q_t, mask)

    refined_rewards = rewards.clone().detach()
    refined_rewards[-1] -= kl_div

    if return_kl_divs:
        return refined_rewards, kl_div
    else:
        return refined_rewards


def compute_kl_div_estimation(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """用无偏估计近似 KL，而不是对完整分布求和。

    E(KL(p || q)) = (q / p) - 1 - log(q / p)，见 GRPO 论文公式 (4)。注意 p 与 q 不要写反。p 需要梯度。

    Args:
        p (torch.Tensor): 参考分布，shape 为 [top_k]，需要梯度。
        q (torch.Tensor): 预测分布，shape 为 [top_k]。
    Returns:
        kl_div (torch.Tensor): 逐位置的无偏估计，shape 为 [top_k]。
    """
    r = q / p
    kl_div = r - 1 - torch.log(r)
    return kl_div


class SlidingWindowStandardization:
    """在固定长度的滑动窗口上在线维护均值和无偏方差。

    传入数据须为 list[float]，避免梯度泄漏以及 float16 的精度和范围问题。
    """
    def __init__(self, max_window_size: int):
        self.max_window_size = max_window_size
        self.raw_data_buffer = deque()
        self.episode_data_sum_buffer = deque()
        self.episode_data_square_sum_buffer = deque()

        self.n = 0
        self.x_sum = 0.0
        self.x_square_sum = 0.0

    def _add_one_episode(self, episode_data: list[float]):
        """向统计量中加入一个 episode。"""
        self.raw_data_buffer.append(episode_data)

        x_sum_new = sum(episode_data)
        x_square_sum_new = sum([x ** 2 for x in episode_data])

        self.episode_data_sum_buffer.append(x_sum_new)
        self.episode_data_square_sum_buffer.append(x_square_sum_new)

        self.n += len(episode_data)
        self.x_sum += x_sum_new
        self.x_square_sum += x_square_sum_new

    def _pop_oldest_episode(self):
        """从统计量中移除最旧的一个 episode。"""
        episode_data = self.raw_data_buffer.popleft()
        x_sum_old = self.episode_data_sum_buffer.popleft()
        x_square_sum_old = self.episode_data_square_sum_buffer.popleft()

        self.n -= len(episode_data)
        self.x_sum -= x_sum_old
        self.x_square_sum -= x_square_sum_old

        return True

    def add_data(self, episode_data_list: list[list[float]]):
        """加入若干 episode。窗口已满时先移除最旧的数据。"""
        for episode_data in episode_data_list:
            if len(self.raw_data_buffer) >= self.max_window_size:
                self._pop_oldest_episode()
            self._add_one_episode(episode_data)

    def get_mean(self) -> float:
        return self.x_sum / self.n if self.n > 0 else 0.0

    def get_variance(self) -> float:
        # 无偏估计
        return (self.x_square_sum / (self.n - 1) - self.x_sum ** 2 / (self.n * (self.n - 1))) if self.n > 1 else 0.0

    def get_std(self) -> float:
        return math.sqrt(self.get_variance())

    def reset(self):
        """重置统计量。"""
        self.raw_data_buffer.clear()
        self.episode_data_sum_buffer.clear()
        self.episode_data_square_sum_buffer.clear()

        self.n = 0
        self.x_sum = 0.0
        self.x_square_sum = 0.0

    def __len__(self):
        return len(self.raw_data_buffer)

    def save(self, filepath: str):
        """把当前状态保存到文件。"""
        state = {
            'max_window_size': self.max_window_size,
            'raw_data_buffer': list(self.raw_data_buffer),
            'episode_data_sum_buffer': list(self.episode_data_sum_buffer),
            'episode_data_square_sum_buffer': list(self.episode_data_square_sum_buffer),
            'n': self.n,
            'x_sum': self.x_sum,
            'x_square_sum': self.x_square_sum,
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"Error saving state to {filepath}: {e}")
            raise

    @classmethod
    def load(cls, filepath: str):
        """从文件恢复状态，并返回新实例。"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            required_keys = ['max_window_size', 'raw_data_buffer', 'episode_data_sum_buffer',
                             'episode_data_square_sum_buffer', 'n', 'x_sum', 'x_square_sum']
            if not all(key in state for key in required_keys):
                raise ValueError(f"Loaded state from {filepath} is missing required keys.")

            instance = cls(state['max_window_size'])

            instance.raw_data_buffer = deque(state['raw_data_buffer'])
            instance.episode_data_sum_buffer = deque(state['episode_data_sum_buffer'])
            instance.episode_data_square_sum_buffer = deque(state['episode_data_square_sum_buffer'])
            instance.n = state['n']
            instance.x_sum = state['x_sum']
            instance.x_square_sum = state['x_square_sum']

            return instance
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            raise
        except Exception as e:
            print(f"Error loading state from {filepath}: {e}")
            raise


def send_to_compute_bertscores(reranker_per_action_responses: list[str], answers: list[str], bertscore_server_url: str) -> dict[str, list[float]]:
    """请求服务器计算 BERTScore。"""
    response = requests.post(bertscore_server_url, json={"candidates": reranker_per_action_responses, "references": answers})
    return response.json()


def _get_reward_prompt(gt: str, question: str) -> str:
    """生成 judge 模型使用的 prompt。"""
    return f"""Background Knowledge: {gt}

Question: {question}
Answer based on the background:
"""


def _process_convs_to_judge_model(
    convs: list[list[dict[str, str]]],
    base_url: str,
    judge_model_tokenizer: PreTrainedTokenizerBase,
    judge_model_name: str,
    timeout: int = 30
) -> list[float]:
    """把对话发给 judge 模型并读取分数。

    batch 为 1 时补一条短样本，避开 sglang 单条请求的问题，返回前再去掉该结果。
    """
    payload = {"model": judge_model_name}
    convs_formatted = []
    for conv in convs:
        conv = judge_model_tokenizer.apply_chat_template(conv, tokenize=False)
        if judge_model_tokenizer.bos_token is not None and conv.startswith(judge_model_tokenizer.bos_token):
            conv = conv[len(judge_model_tokenizer.bos_token):]
        convs_formatted.append(conv)

    is_padded = False
    if len(convs_formatted) == 1:
        dummy_conv = [
            {"role": "user", "content": "."},
            {"role": "assistant", "content": "."}
        ]
        dummy_formatted = judge_model_tokenizer.apply_chat_template(dummy_conv, tokenize=False)

        if judge_model_tokenizer.bos_token is not None and dummy_formatted.startswith(judge_model_tokenizer.bos_token):
            dummy_formatted = dummy_formatted[len(judge_model_tokenizer.bos_token):]

        convs_formatted.append(dummy_formatted)
        is_padded = True

    payload.update({"text": convs_formatted})
    rewards = []
    try:
        responses = requests.post(base_url, json=payload, timeout=timeout).json()

        for response in responses:
            if isinstance(response, dict) and "embedding" in response:
                rewards.append(response["embedding"][0])
            else:
                pass

        if is_padded:
            rewards = rewards[:1]

        assert len(rewards) == len(convs), f"Expected {len(convs)} rewards, got {len(rewards)}"
        return rewards
    except Exception as e:
        print(f"Error processing convs to judge model at {base_url}: {e}", flush=True)
        return [None] * len(convs)


def send_to_judge_model_to_get_scores(
        reranker_per_action_responses: list[str],
        answers: list[str],
        query: str,
        judge_model_server_urls: list[str],
        world_size: int,
        rank: int,
        judge_model_tokenizer: PreTrainedTokenizerBase,
        judge_model_name: str,
        timeout: int = 30
    ) -> list[float]:
    """把每个 action 的回答发给 judge 模型，返回分数。

    world_size 能被服务器数量整除时，同一 rank 固定请求一台；否则逐条轮询。
    服务接收 chat 文本，分数在返回的 embedding 中。
    """
    assert len(answers) == len(reranker_per_action_responses), f"The number of answers and reranker_per_action_responses must be the same. now lengths: {len(answers)} and {len(reranker_per_action_responses)}"

    convs = [
        [
            {"role": "user", "content": _get_reward_prompt(answers[i], query)},
            {"role": "assistant", "content": reranker_per_action_responses[i]}
        ]
        for i in range(len(reranker_per_action_responses))
    ]

    server_count = len(judge_model_server_urls)
    if world_size % server_count == 0:
        target_url = judge_model_server_urls[rank % server_count]
        scores = _process_convs_to_judge_model(convs, target_url, judge_model_tokenizer, judge_model_name, timeout)
    else:
        scores = []
        for i in range(len(convs)):
            target_url = judge_model_server_urls[i % server_count]
            single_conv_scores = _process_convs_to_judge_model([convs[i]], target_url, judge_model_tokenizer, judge_model_name, timeout)
            scores.append(single_conv_scores[0])

    return scores


def send_to_judge_model_to_get_scores_single_case(
    response: str,
    ground_truths: list[str],
    query: str,
    judge_model_server_urls: list[str],
    judge_model_tokenizer: PreTrainedTokenizerBase,
    judge_model_name: str,
    timeout: int = 30
) -> dict[str, float]:
    """把单条回答发给 judge 模型，返回分数字典。多台服务器按输入做稳定分配。"""
    ground_truths_str = "\n".join([f"{i+1}. {gt}" for i, gt in enumerate(ground_truths)])
    prompt = _get_reward_prompt(ground_truths_str, query)

    conv = [
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    ]

    server_count = len(judge_model_server_urls)
    server_idx = hash(query + response) % server_count
    target_url = judge_model_server_urls[server_idx]

    scores = _process_convs_to_judge_model(conv, target_url, judge_model_tokenizer, judge_model_name, timeout)
    score = scores[0] if scores else None

    return {"judge_model_score": score}
