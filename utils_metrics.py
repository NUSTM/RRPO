import torch

from collections import Counter

import re
import string

from rouge import Rouge
from nltk import PorterStemmer

from copy import deepcopy
from tqdm import tqdm

from typing import Union


# 以下文本归一化来自 HotpotQA 官方评估脚本。
def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

def white_space_fix(text):
    return ' '.join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def lower(text):
    return text.lower()


def normalize_answer(s):
    """HotpotQA 官方归一化：小写、去标点、去冠词、合并空白。"""
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(normalized_prediction, normalized_ground_truth):
    """计算已归一化文本的 token 级 F1。"""
    ZERO_METRIC = 0.0

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def exact_match_score(normalized_prediction, normalized_ground_truth):
    """判断已归一化文本是否完全一致。"""
    return normalized_prediction == normalized_ground_truth

def hit_score(normalized_prediction, normalized_ground_truth):
    """答案包含标准答案时返回 1，否则返回 -1。"""
    if normalized_ground_truth in normalized_prediction:
        return 1.0
    else:
        return -1.0


def compute_em_f1_from_reader_response(reader_response: str, answer: str) -> tuple[float, float]:
    """计算 EM 和 F1。"""
    normalized_sampled_output = normalize_answer(reader_response)
    normalized_ground_truth = normalize_answer(answer)
    em = exact_match_score(normalized_sampled_output, normalized_ground_truth)
    f1 = f1_score(normalized_sampled_output, normalized_ground_truth)
    return em, f1


def compute_em_f1_from_reader_response_for_multi_answers(reader_response: str, answers: list[str]) -> tuple[float, float]:
    """计算 EM 和 F1。一个问题有多个答案时，对每个答案分别计算后取最大值。"""
    normed_sampled_output = normalize_answer(reader_response)
    normed_ground_truths = [normalize_answer(s) for s in answers]

    em_scores, f1_scores = [], []
    for normed_ground_truth in normed_ground_truths:
        em = exact_match_score(normed_sampled_output, normed_ground_truth)
        f1 = f1_score(normed_sampled_output, normed_ground_truth)

        em_scores.append(em)
        f1_scores.append(f1)

    em = max(em_scores)
    f1 = max(f1_scores)

    return em, f1


def compute_reward(sampled_output: list[str], ground_truth: list[str], f1_lambda: float, hit_lambda: float, device: torch.device = None, dtype = None) -> torch.Tensor:
    """用 EM、F1 和 hit 计算 reward。"""
    assert len(sampled_output) == len(ground_truth), "sampled_output和ground_truth的长度不一致"

    normalized_sampled_output = [normalize_answer(s) for s in sampled_output]
    normalized_ground_truth = [normalize_answer(s) for s in ground_truth]

    rewards = []
    for normalized_pred, normalized_target in zip(normalized_sampled_output, normalized_ground_truth):
        em = exact_match_score(normalized_pred, normalized_target)
        f1 = f1_score(normalized_pred, normalized_target)
        hit = hit_score(normalized_pred, normalized_target)

        reward = em + f1_lambda * f1 + hit_lambda * hit
        rewards.append(reward)

    return torch.tensor(rewards, dtype=dtype if dtype else torch.float32, device=torch.device('cuda') if device is None else device)


def compute_episode_rewards_for_multi_answers(sampled_output: list[str], ground_truths: list[str], f1_lambda: float, hit_lambda: float, device: torch.device = None, dtype = None) -> torch.Tensor:
    """用 EM、F1 和 hit 计算 reward。多个答案只提供一遍，对每个答案分别计算后取最大值。"""
    normalized_sampled_output = [normalize_answer(s) for s in sampled_output]
    normalized_ground_truths = [normalize_answer(s) for s in ground_truths]

    rewards = []
    for normalized_pred in normalized_sampled_output:
        em_scores, f1_scores, hit_scores = [], [], []
        for normed_ground_truth in normalized_ground_truths:
            em = exact_match_score(normalized_pred, normed_ground_truth)
            f1 = f1_score(normalized_pred, normed_ground_truth)
            hit = hit_score(normalized_pred, normed_ground_truth)

            em_scores.append(em)
            f1_scores.append(f1)
            hit_scores.append(hit)

        em = max(em_scores)
        f1 = max(f1_scores)
        hit = max(hit_scores)

        reward = em + f1_lambda * f1 + hit_lambda * hit
        rewards.append(reward)

    return torch.tensor(rewards, dtype=dtype if dtype else torch.float32, device=torch.device('cuda') if device is None else device)


def compute_ave_em_f1_from_reader_results(reader_results: dict[str, dict[str, any]]) -> tuple[float, float]:
    """计算所有 case 的平均 EM 和 F1。"""
    em_scores = [v["em"] for v in reader_results.values()]
    f1_scores = [v["f1"] for v in reader_results.values()]
    em_ave = sum(em_scores) / len(em_scores)
    f1_ave = sum(f1_scores) / len(f1_scores)
    return em_ave, f1_ave


def compute_ave_rouge_from_reader_results(reader_results: dict[str, dict[str, any]], return_rouge_L_f1_only: bool) -> float:
    """计算所有 case 的平均 ROUGE。return_rouge_L_f1_only 为 True 时只返回 rouge-l 的 F1。"""
    count = len(reader_results)
    first_value = next(iter(reader_results.values()))
    metric_result = deepcopy(first_value)
    del metric_result['response']
    for k1 in metric_result.keys():
        for k2 in metric_result[k1].keys():
            tmp = [v[k1][k2] for v in reader_results.values()]
            metric_result[k1][k2] = sum(tmp) / count

    if return_rouge_L_f1_only:
        return metric_result['rouge-l']['f']
    else:
        return metric_result


class RougeMetric():

    def __init__(self):
        self.rouge = Rouge()
        self.stemmer = PorterStemmer()

    def compute_scores(self, hypotheses: list[str], references: list[str], return_rouge_L_f1_only: bool, avg: bool, show_progress_bar: bool = False):
        """批量计算 ROUGE。hypotheses 和 references 都是字符串列表。"""
        if show_progress_bar:
            cleaned_hypoth = [" ".join([self.stemmer.stem(i) for i in line.split()]) for line in tqdm(hypotheses)]
            cleaned_ref = [" ".join([self.stemmer.stem(i) for i in line.split()]) for line in tqdm(references)]
        else:
            cleaned_hypoth = [" ".join([self.stemmer.stem(i) for i in line.split()]) for line in hypotheses]
            cleaned_ref = [" ".join([self.stemmer.stem(i) for i in line.split()]) for line in references]

        scores = self.rouge.get_scores(cleaned_hypoth, cleaned_ref, avg=avg)

        if return_rouge_L_f1_only:
            if avg:
                return scores['rouge-l']['f']
            else:
                return [score['rouge-l']['f'] for score in scores]
        else:
            return scores

    def compute_score_single_case(self, hypothesis: str, reference: str, return_rouge_L_f1_only: bool) -> Union[float, dict[str, dict[str, float]]]:
        """计算单条 ROUGE。hypothesis 和 reference 都是字符串。"""
        cleaned_hypoth = " ".join([self.stemmer.stem(i) for i in hypothesis.split()])
        cleaned_ref = " ".join([self.stemmer.stem(i) for i in reference.split()])
        scores = self.rouge.get_scores([cleaned_hypoth], [cleaned_ref])
        scores = scores[0]

        if return_rouge_L_f1_only:
            return scores['rouge-l']['f']
        else:
            return scores
