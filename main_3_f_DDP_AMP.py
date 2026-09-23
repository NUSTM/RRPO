# -*- coding: utf-8 -*-
"""
Distributed PPO training for reranking (DDP + AMP).

Launch with torchrun. Select a config via --env ENV1 or ENV2.
Optional flags: --debug_using_small_data, --resume_from_checkpoint, --not_using_sql.
"""

from CFG_2 import MetricSetting

# Default seed. seed_generator starts at 0. A non-default seed makes seed_generator start from that value.
# Used by set_seed, torch.manual_seed, and DistributedSampler. seed_generator is seeded separately.
SEED: int = 42
import argparse
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env', type=str, help='ENV1 or ENV2.')
    parser.add_argument('--debug_using_small_data', action='store_true', help='Whether to use small data for debugging.')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Whether to resume from checkpoint.')
    parser.add_argument('--not_using_sql', action='store_true', help='Whether to not use sql.')

    args = parser.parse_args()
    return args

args = parse_args()
if args.env == "ENV1":
    from CFG_2 import Main_CFG_ENV1 as CFG
elif args.env == "ENV2":
    from CFG_2 import Main_CFG_ENV2 as CFG
else:
    raise ValueError(f"Invalid environment: {args.env}")

METRIC_SETTING = CFG.METRIC_SETTING
MODEL_SETTING = CFG.MODEL_SETTING

DEBUG_USING_SMALL_DATA = args.debug_using_small_data

RESUME_FROM_CHECKPOINT = args.resume_from_checkpoint

USING_SQL = not args.not_using_sql

"""设置reader server地址"""
READER_SERVER_URLS = CFG.reader_server_urls


import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from openai import OpenAI

import multiprocessing
import threading

from copy import deepcopy
import math
import time
from tqdm import tqdm
import pickle
import shutil
import os
import traceback

import datetime

import itertools
import collections

from typing import Optional

if USING_SQL:
    import psycopg2

import re


# ------ ------ self constructed ------ ------
if MODEL_SETTING == "gte":
    from src.models import MyModel_GTE as MyModel
elif MODEL_SETTING == "jina":
    from src.models import MyModel_JINA as MyModel
elif MODEL_SETTING == "bge":
    from src.models import MyModel_BGE as MyModel
elif MODEL_SETTING == "qwen3_reranker":
    from src.models import MyModel_Qwen3_Reranker as MyModel
else:
    raise NotImplementedError("MODEL_SETTING is supported to be 'gte', 'jina', 'bge', 'qwen3_reranker'.")

from src.my_datasets_2 import (
    FlexibleQADataset, FlexibleQADataset_JsonlLazy
)
if METRIC_SETTING == MetricSetting.MULTIHOPQA:
    from src.my_datasets_2 import multihop_qa_data_parser as dataset_parser
elif METRIC_SETTING == MetricSetting.ELI5:
    from src.my_datasets_2 import eli5_data_parser as dataset_parser
elif METRIC_SETTING == MetricSetting.AMBIGNQ:
    from src.my_datasets_2 import ambig_nq_data_parser as dataset_parser
elif METRIC_SETTING == MetricSetting.MUSIQUE:
    from src.my_datasets_2 import musique_data_parser as dataset_parser
elif METRIC_SETTING == MetricSetting.ASQA:
    from src.my_datasets_2 import asqa_data_parser as dataset_parser
else:
    raise NotImplementedError(f"METRIC_SETTING: {METRIC_SETTING} is not supported.")


if DEBUG_USING_SMALL_DATA:
    TRAIN_SMALL_DATA_LIMIT = 50
    TEST_SMALL_DATA_LIMIT = 100


from utils import (
    select_max_prob_action,
    set_seed,
    save_results,
    compute_logits,
    compute_single_case_logits,
    sort_rerank_result_and_get_action_passages,

    # for PPO
    freeze_model,
    get_action_log_prob,
    compute_kl_div_estimation,

    load_results,
    SeedGenerator,
    compute_GAE,
    redirect_output,
    compute_ave_em_f1_from_reader_results,
    compute_ave_rouge_from_reader_results,
    mycollate_fn,

    SlidingWindowStandardization,
    print_time,

    get_episode_v_responses_from_cache,
)

if not USING_SQL:
    from utils_sending_response_no_sql import (
        get_episode_reader_response_with_rank_check,
    )
    from eval_process_tools_3_no_sql import (
        Shared_Variables,
        inference_worker,
        monitor_and_send,
    )
else:
    from utils_sending_response_using_sql import (
        get_episode_reader_response_with_rank_check,
    )
    from eval_process_tools_3_using_sql import (
        Shared_Variables,
        inference_worker,
        monitor_and_send,
    )

USING_ROUGE_F1_AND_BERTSCORE: bool = False

if METRIC_SETTING in {MetricSetting.MULTIHOPQA, MetricSetting.MUSIQUE}:
    from utils import compute_reward
elif METRIC_SETTING in {MetricSetting.ELI5}:
    from utils import RougeMetric
elif METRIC_SETTING in {MetricSetting.ASQA}:
    if USING_ROUGE_F1_AND_BERTSCORE == True:
        from utils import RougeMetric
elif METRIC_SETTING == MetricSetting.AMBIGNQ:
    from utils import compute_episode_rewards_for_multi_answers
else:
    raise NotImplementedError(f"METRIC_SETTING: {METRIC_SETTING} is not supported.")


if METRIC_SETTING == MetricSetting.ASQA:
    if USING_ROUGE_F1_AND_BERTSCORE == True:
        if not hasattr(CFG, 'bertscore_server_url'):
            raise ValueError("You set METRIC_SETTING to MetricSetting.ASQA, then CFG.bertscore_server_url must be set.")
        from utils import send_to_compute_bertscores
    else:
        if not hasattr(CFG, 'judge_model_server_urls'):
            raise ValueError("You set METRIC_SETTING to MetricSetting.ASQA, then CFG.judge_mode_server_urls must be set.")
        from utils import send_to_judge_model_to_get_scores


from eval_dist_sampler import EvalDistributedSampler



GPUS_PER_RANK = 1

os.environ['TORCH_NCCL_BLOCKING_WAIT'] = "1"


ALLOW_REFERENCE_MODEL_UPDATE: bool = False
USE_SAVED_REFERENCE_MODEL_RERANK_RESULTS: bool = False
USE_SAVED_REFERENCE_MODEL_READER_RESULTS: bool = False
USE_SAVED_REFERENCE_MODEL_METRIC_RESULTS: bool = False



"""------ reader服务器设置 ------"""
clients = [
    OpenAI(
        api_key="EMPTY",
        base_url=url,
        timeout=1200,
    ) for url in READER_SERVER_URLS
]


"""------ 数据库连接参数 ------"""
if USING_SQL:
    from CFG_2 import SQL_CFG
    SQL_DB_PARAMS = SQL_CFG.SQL_DB_PARAMS


assert hasattr(CFG, 'main_version') and CFG.main_version == "3f_DDP_AMP", f"CFG.main_version: {CFG.main_version}, but you are using **main_3_f_DDP_AMP.py**. if you insist to use this version, please set CFG.main_version to '3f_DDP_AMP', or forbid this code to run."

assert hasattr(CFG, 'train_version') and hasattr(CFG, "save_path") and hasattr(CFG, "DDP_log_filename_prefix")


if MODEL_SETTING in {"gte", "jina"}:
    if not hasattr(CFG, 'additional_file_path') and hasattr(CFG, 'additional_file_names'):
        raise ValueError(f"You set MODEL_SETTING to {MODEL_SETTING}, then CFG.additional_file_path and CFG.additional_file_names must be set.")

if RESUME_FROM_CHECKPOINT:
    if not hasattr(CFG, 'resume_checkpoint_path'):
        raise ValueError("You set RESUME_FROM_CHECKPOINT to True, then CFG.resume_checkpoint_path must be set.")
    if not hasattr(CFG, 'resume_reference_model_path'):
        raise ValueError("You set RESUME_FROM_CHECKPOINT to True, then CFG.resume_reference_model_path must be set.")

if not ALLOW_REFERENCE_MODEL_UPDATE:
    if not hasattr(CFG, 'save_steps'):
        raise ValueError("You set ALLOW_REFERENCE_MODEL_UPDATE to False, then CFG.save_steps must be set.")

if USE_SAVED_REFERENCE_MODEL_RERANK_RESULTS:
    if not hasattr(CFG, "reference_logits_datapath"):
        raise ValueError("You set USE_SAVED_REFERENCE_MODEL_RERANK_RESULTS to True, then CFG.reference_logits_datapath must be set.")

if USE_SAVED_REFERENCE_MODEL_READER_RESULTS:
    if not hasattr(CFG, "reader_results_bm25_reranker_raw_datapath"):
        raise ValueError("You set USE_SAVED_REFERENCE_MODEL_READER_RESULTS to True, then CFG.reader_results_bm25_reranker_raw_datapath must be set.")

if USE_SAVED_REFERENCE_MODEL_METRIC_RESULTS:
    if not hasattr(CFG, "init_reference_eval_results"):
        raise ValueError("You set USE_SAVED_REFERENCE_MODEL_METRIC_RESULTS to True, then CFG.init_reference_eval_results must be set.")
    

if DEBUG_USING_SMALL_DATA:
    INIT_EVAL_FINISH_COUNT_CHECK_INTERVAL = 10
    SYNC_RERANK_RESULTS_INTERVAL = 5

    if not ALLOW_REFERENCE_MODEL_UPDATE:
        CFG.save_steps = 10
    else:
        CFG.check_for_reference_model_steps = 10
else:
    INIT_EVAL_FINISH_COUNT_CHECK_INTERVAL = 1000
    SYNC_RERANK_RESULTS_INTERVAL = 50


if USING_SQL:
    conn = None


set_seed(SEED)




def _get_v_responses_no_sql(
        all_logits_reference: torch.Tensor, 
        query: str, 
        passages: list[str],
        llm_reader_name: str,
        prompt_template: str,
        rank: int,
        world_size: int,
        top_k: int,
        ):
    """不使用SQL时的V(s)计算逻辑"""
    rerank_result = {"scores": all_logits_reference, "passages": passages}
    sorted_rerank_result, action_passages = sort_rerank_result_and_get_action_passages(rerank_result, top_k)

    reranker_per_action_responses = get_episode_reader_response_with_rank_check(
        query, action_passages, llm_reader_name, prompt_template, clients, rank, world_size, METRIC_SETTING
    )

    return reranker_per_action_responses

def _get_v_responses_sql(
        all_logits_reference: torch.Tensor, 
        query: str, 
        passages: list[str],
        case_id: str,
        passage_ids: list[str],
        conn: psycopg2.extensions.connection,
        cur: psycopg2.extensions.cursor,
        llm_reader_name: str,
        prompt_template: str,
        rank: int,
        world_size: int,
        gpu_type: str,
        top_k: int,
        ):
    """使用SQL时的V(s)计算逻辑"""
    rerank_result = {"scores": all_logits_reference, "passages": passages, "passage_ids": passage_ids}
    sorted_rerank_result, action_passages = sort_rerank_result_and_get_action_passages(rerank_result, top_k)
    action_passage_ids = sorted_rerank_result["passage_ids"][:top_k]

    reranker_per_action_responses = get_episode_reader_response_with_rank_check(
        query=query,
        action_passages=action_passages,
        case_id=case_id,
        action_passage_ids=action_passage_ids,
        conn=conn,
        cur=cur,
        case_type="train_v_response",
        gpu_type=gpu_type,
        llm_reader_name=llm_reader_name,
        prompt_template=prompt_template,
        clients=clients,
        rank=rank,
        world_size=world_size,
        metric_setting=METRIC_SETTING,
    )
    
    return reranker_per_action_responses

if not USING_SQL:
    get_episode_v_responses_with_rank_check = _get_v_responses_no_sql
else:
    get_episode_v_responses_with_rank_check = _get_v_responses_sql

get_episode_v_responses_with_rank_check.__doc__ = (
    "State value V(s) from the reference reranker's top-k passages."
)


def set_save_path(epoch: int, global_step: int = None) -> str:
    if global_step == None:
        save_path = f"{CFG.save_path}{epoch+1}epoch"
    else:
        save_path = f"{CFG.save_path}{epoch+1}epoch_step_{global_step}"
    return save_path


def save_special_model(
        ddp_model: nn.Module, 
        additional_file_path: str,
        additional_file_names: list[str],
        epoch: int, 
        global_step: int = None, 
        return_savepath: bool = False
    ) -> Optional[str]:
    
    # 设置路径
    save_path = set_save_path(epoch, global_step)
    if global_step == None:
        info_str = f"Saved checkpoint at epoch {epoch+1}."
    else:
        info_str = f"Saved checkpoint at epoch {epoch+1}, step {global_step}."

    # 解包 DDP 模型
    unwrapped_model = ddp_model.module
    # 调用 Hugging Face 的 save_pretrained 方法
    unwrapped_model.model.save_pretrained(save_path)
    print(info_str, flush=True)

    copy_additional_files(additional_file_path, additional_file_names, save_path)

    if return_savepath:
        return save_path
    

def copy_additional_files(
        additional_file_path: str,
        additional_file_names: list[str],
        save_path: str
    ) -> None:
    """复制额外的文件到保存路径"""
    for file_name in additional_file_names:
        # 构建完整的源文件路径和目标文件路径
        source_file = os.path.join(additional_file_path, file_name)
        destination_file = os.path.join(save_path, file_name)
        # 复制文件
        shutil.copy2(source_file, destination_file)


def save_model(
        ddp_model: nn.Module, 
        epoch: int, 
        global_step: int = None, 
        return_savepath: bool = False
    ) -> Optional[str]:
    
    # 设置路径
    save_path = set_save_path(epoch, global_step)
    if global_step == None:
        info_str = f"Saved checkpoint at epoch {epoch+1}."
    else:
        info_str = f"Saved checkpoint at epoch {epoch+1}, step {global_step}."

    # 解包 DDP 模型
    unwrapped_model = ddp_model.module
    # 调用 Hugging Face 的 save_pretrained 方法
    unwrapped_model.model.save_pretrained(save_path)
    print(info_str, flush=True)

    if return_savepath:
        return save_path

    
def save_checkpoint(epoch: int, step: int, optimizer, scheduler, scaler, checkpoint_dir: str, filename: str = "checkpoint.pt", is_best: bool = False):
    """
    保存检查点，仅由 Rank 0 执行。
    包含模型、优化器、调度器、AMP Scaler 状态以及 epoch/step。
    
    Args:
        epoch (int): 当前完成的 epoch 数。
        step (int): 当前完成的总训练步数。
        optimizer (torch.optim.Optimizer): 优化器实例。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器实例 (可以为 None)。
        scaler (torch.cuda.amp.GradScaler): GradScaler 实例。
        checkpoint_dir (str): 保存检查点的目录。
        filename (str): 检查点文件名 (例如 'checkpoint.pt' 或 'latest.pt')。
        is_best (bool): 是否为当前最佳模型，用于额外保存 best.pt。
    """
    rank = dist.get_rank()
    if rank == 0:
        # 确保目录存在
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 准备需要保存的状态字典
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }


        # 原子性保存：先保存到临时文件
        temp_filepath = os.path.join(checkpoint_dir, f"temp_{filename}")
        final_filepath = os.path.join(checkpoint_dir, filename)

        print(f"Rank 0: Saving checkpoint to {final_filepath} (Epoch: {epoch}, Step: {step})...")
        try:
            torch.save(checkpoint, temp_filepath)
            # 保存成功后，重命名为最终文件 (原子操作)
            shutil.move(temp_filepath, final_filepath)
            print(f"Rank 0: Checkpoint saved successfully to {final_filepath}")

            # 如果是最佳模型，额外复制一份
            if is_best:
                best_filepath = os.path.join(checkpoint_dir, "best_model.pt")
                shutil.copyfile(final_filepath, best_filepath)
                print(f"Rank 0: Copied best model checkpoint to {best_filepath}")

        except Exception as e:
            print(f"Rank 0: Error saving checkpoint: {e}")
            # 1. 获取完整的 traceback 字符串
            tb_str = traceback.format_exc()
            # 2. 准备包含上下文的详细错误消息
            error_message = (
                f"==================== [ERROR] in compute_reward ====================\n"
                f"METRIC_SETTING: {METRIC_SETTING}\n"
                f"Exception Type: {type(e).__name__}\n"
                f"Exception Details: {e}\n"
                f"----------------------- Traceback -----------------------\n"
                f"{tb_str}\n"
                f"================================================================="
            )
                
            # 3. 打印完整的错误信息
            print(error_message, flush=True)

            if os.path.exists(temp_filepath):
                os.remove(temp_filepath) # 清理临时文件


def load_checkpoint(checkpoint_path, optimizer, scheduler, scaler, local_rank) -> tuple[int, int]:
    """
    加载检查点，所有进程都需要执行。
    
    Args:
        checkpoint_path (str): 检查点文件路径。
        optimizer (torch.optim.Optimizer): 优化器实例。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器实例 (可以为 None)。
        scaler (torch.cuda.amp.GradScaler): GradScaler 实例。
        local_rank (int): 当前进程的本地 GPU 编号。

    Returns:
        tuple: (start_epoch, start_step) 从检查点恢复的 epoch 和 step。
    """
    map_location = 'cpu'
    
    print(f"Rank {dist.get_rank()}: Loading checkpoint from {checkpoint_path} with map_location='{map_location}'...")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0) # 使用 get 提供默认值
    start_step = checkpoint.get('step', 0) # 使用 get 提供默认值
    
    print(f"Rank {dist.get_rank()}: Checkpoint loaded. Resuming from Epoch {start_epoch}, Step {start_step}.")

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(local_rank)

    return start_epoch, start_step


def save_eval_results(
        rerank_results_dic_all: dict[str, any], 
        reader_results_dic_all: dict[str, any], 
        eval_metric_results: dict[str, any], 
        model_savepath: str,
        epoch: int,
        global_step: int
        ):
    """保存评估的结果: rerank结果, reader结果, metric结果"""
    # 先创建目录
    eval_results_savepath = model_savepath + "/eval_results"
    if not os.path.exists(eval_results_savepath):
        os.makedirs(eval_results_savepath)
    # 保存结果为文件
    savepath_appendix = f"{epoch+1}epoch_step_{global_step}.json"
    save_results(rerank_results_dic_all, eval_results_savepath + f"/rerank_results_{savepath_appendix}")
    save_results(reader_results_dic_all, eval_results_savepath + f"/reader_results_{savepath_appendix}")
    save_results(eval_metric_results, eval_results_savepath + f"/metric_results_{savepath_appendix}")


def train(rank: int, local_rank: int, world_size: int):

    """读取全局变量"""
    global USE_SAVED_REFERENCE_MODEL_RERANK_RESULTS, USE_SAVED_REFERENCE_MODEL_READER_RESULTS
    ACCUMULATION_STEPS_PER_RANK = CFG.accumulation_steps_per_rank


    dist.barrier()
    # 只在rank0创建进程和子进程/子线程“安全”共享的变量
    if rank == 0:
        shared_variables = Shared_Variables()
        # 1) 创建 manager
        manager = multiprocessing.Manager()

        # 2) 创建 Shared_Variables 实例，并初始化(部分初始化)
        shared_variables = Shared_Variables()
        shared_variables.init_with_manager(manager)

    dist.barrier()


    torch.manual_seed(SEED)

    self_device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    print_with_rank_check(f'rank: {rank}, self_device: {self_device}', flush=True)

    if CFG.AMP_dtype == 'float16':
        AMP_DTYPE = torch.float16
    elif CFG.AMP_dtype == 'bfloat16':
        AMP_DTYPE = torch.bfloat16
    else:
        raise ValueError("AMP_dtype must be 'float16' or 'bfloat16'.")
    
    print_with_rank_check(f'AMP_DTYPE: {AMP_DTYPE}', flush=True)

    if USE_SAVED_REFERENCE_MODEL_READER_RESULTS:
        reader_results_bm25_reranker_raw = load_results(CFG.reader_results_bm25_reranker_raw_datapath)
    if USE_SAVED_REFERENCE_MODEL_RERANK_RESULTS:
        reference_logits_data = load_results(CFG.reference_logits_datapath)

    seed_generator = SeedGenerator()
    if SEED != 42:
        seed_generator.set_seed(SEED)

    self_gpu_type_raw = torch.cuda.get_device_name(local_rank)
    if self_gpu_type_raw == "NVIDIA A800 80GB PCIe":
        self_gpu_type = "A800"
    else:
        self_gpu_type = re.sub(r"^(?:NVIDIA\s+|GeForce\s+|RTX\s+)+", "", self_gpu_type_raw)

    print_with_rank_check(f'self_gpu_type: {self_gpu_type}', flush=True)


    """初始化tokenizer"""
    if MODEL_SETTING not in {"qwen3_reranker"}:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(CFG.model_path, use_fast=True)
    else:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            CFG.model_path, 
            use_fast=True,
            padding_side='left',
        )

    """------ 加载 actor 模型并用DDP包装 ------"""
    load_path = CFG.model_path if not RESUME_FROM_CHECKPOINT else CFG.resume_checkpoint_path
    if MODEL_SETTING not in {"qwen3_reranker"}:
        model = MyModel(model_path=load_path, device=self_device)
    else:
        model = MyModel(
            model_path=load_path, 
            tokenizer=tokenizer,
            device=self_device
        )

    self_dtype = model.model.dtype

    if CFG.use_gradient_checkpointing:
        model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
        if model.model.config.is_encoder_decoder or hasattr(model.model.config, "use_cache"):
            print("Setting use_cache to False for gradient checkpointing in actor model.")
            model.model.config.use_cache = False

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    print_with_rank_check(f'Actor model created and wrapped in DDP. self_dtype: {self_dtype}', flush=True)


    """------ 加载 reference 模型并冻结参数 ------"""
    reference_load_path = CFG.model_path if not RESUME_FROM_CHECKPOINT else CFG.resume_reference_model_path
    if MODEL_SETTING not in {"qwen3_reranker"}:
        reference_model = MyModel(model_path=reference_load_path, device=self_device)
    else:
        reference_model = MyModel(
            model_path=reference_load_path, 
            tokenizer=tokenizer,
            device=self_device
        )

    reference_model.eval()
    freeze_model(reference_model)
    
    print_with_rank_check(f'Reference model created and parameters frozen.', flush=True)

    # BF16 does not use GradScaler; the unscale kernel is not implemented for BFloat16.
    if MODEL_SETTING not in {"qwen3_reranker"}:
        scaler = torch.amp.GradScaler('cuda', enabled=CFG.use_amp)
    else:
        use_scaler = (CFG.AMP_dtype == 'float16')
        scaler = torch.amp.GradScaler('cuda', enabled=(CFG.use_amp and use_scaler))

    # 定义数据集和其他组件
    print(f"try to load train_dataset...", flush=True)
    if CFG.bm25_data_path.endswith(".jsonl"):
        my_dataset = FlexibleQADataset_JsonlLazy(
            raw_data_path=CFG.train_data_path,
            bm25_data_path=CFG.bm25_data_path,
            tokenizer=tokenizer,
            max_seq_length=CFG.max_seq_length,
            data_parser=dataset_parser,
            limit=TRAIN_SMALL_DATA_LIMIT if DEBUG_USING_SMALL_DATA else None,
        )
    else:
        my_dataset = FlexibleQADataset(
            raw_data_path=CFG.train_data_path,
            bm25_data_path=CFG.bm25_data_path,
            tokenizer=tokenizer,
            max_seq_length=CFG.max_seq_length,
            data_parser=dataset_parser,
            limit=TRAIN_SMALL_DATA_LIMIT if DEBUG_USING_SMALL_DATA else None,
        )
    print(f"train dataset loaded.", flush=True)
    print(f"trying to load train sampler...", flush=True)
    sampler = DistributedSampler(
        my_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True, 
        seed=SEED
    )
    print(f"train sampler loaded.", flush=True)
    print(f"trying to load train dataloader...", flush=True)
    my_dataloader = DataLoader(
        my_dataset,
        batch_size=1,
        collate_fn=mycollate_fn,
        sampler=sampler,
    )
    print(f"train dataloader loaded.", flush=True)

    print(f"trying to load test_dataset...", flush=True)
    test_dataset = FlexibleQADataset(
        raw_data_path=CFG.test_data_path,
        bm25_data_path=CFG.test_bm25_data_path,
        tokenizer=tokenizer,
        max_seq_length=CFG.max_seq_length,
        data_parser=dataset_parser,
        limit=TEST_SMALL_DATA_LIMIT if DEBUG_USING_SMALL_DATA else None,
    )
    test_sampler = EvalDistributedSampler(
        test_dataset, 
        num_replicas=world_size,
        rank=rank, 
        shuffle=False, 
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=mycollate_fn,
        sampler=test_sampler,
    )

    print_with_rank_check(f'dataset and dataloader init finished.')
    print_with_rank_check(f'train dataset len: {len(my_dataset)}')
    print_with_rank_check(f"test dataset len: {len(test_dataset)}")

    optimizer = AdamW(
        [
            {
                'params': ddp_model.parameters(),
                'lr': CFG.learning_rate
            },
        ],
    )
    scheduler = LinearLR(
        optimizer=optimizer,
        start_factor=CFG.scheduler_start_factor,
        end_factor=CFG.scheduler_end_factor,
        total_iters=CFG.training_epochs * math.ceil(len(my_dataloader) / ACCUMULATION_STEPS_PER_RANK),
    )

    if RESUME_FROM_CHECKPOINT:
        resume_ckpt_dir = os.path.join(CFG.resume_checkpoint_path, "checkpoint.pt")
        resume_start_epoch, resume_start_global_step = load_checkpoint(
            resume_ckpt_dir, optimizer, scheduler, scaler, local_rank
        )

        # Assumes the dataloader length is unchanged across resume.
        steps_done_in_previous_epochs = resume_start_epoch * len(my_dataloader)
        resume_start_step_total = resume_start_global_step // world_size
        resume_start_step = int(resume_start_step_total - steps_done_in_previous_epochs)

        if SEED == 42:
            resume_start_seed = int(resume_start_global_step / world_size)
        else:
            resume_start_seed = SEED + int(resume_start_global_step / world_size)
        seed_generator.set_seed(resume_start_seed)

        print_with_rank_check(f"RESUME INFO: Start Epoch: {resume_start_epoch}, Skip Steps: {resume_start_step}, Global Step: {resume_start_global_step}")
    
    print_with_rank_check(f'optimizer and scheduler loaded.', flush=True)

    if rank == 0:
        shared_variables.eval_task_count = len(test_dataset)


    if USE_SAVED_REFERENCE_MODEL_METRIC_RESULTS:
        if not RESUME_FROM_CHECKPOINT:
            reference_eval_results = deepcopy(CFG.init_reference_eval_results)
        else:
            raise NotImplementedError("Resuming with saved reference metric results is not implemented.")

    if ALLOW_REFERENCE_MODEL_UPDATE:
        if not RESUME_FROM_CHECKPOINT:
            check_for_reference_model_interval: int = CFG.check_for_reference_model_steps
        else:
            if resume_start_global_step % CFG.check_for_reference_model_steps == 0:
                check_for_reference_model_interval: int = resume_start_global_step + CFG.check_for_reference_model_steps
            else:
                check_for_reference_model_interval: int = math.ceil(resume_start_global_step / CFG.check_for_reference_model_steps) * CFG.check_for_reference_model_steps
                check_for_reference_model_interval = int(check_for_reference_model_interval)

        print_with_rank_check(f"check_for_reference_model_interval: {check_for_reference_model_interval}", flush=True)
    
    else:
        if not RESUME_FROM_CHECKPOINT:
            save_interval: int = CFG.save_steps
        else:
            if resume_start_global_step % CFG.save_steps == 0:
                save_interval: int = resume_start_global_step + CFG.save_steps
            else:
                save_interval: int = math.ceil(resume_start_global_step / CFG.save_steps) * CFG.save_steps
                save_interval = int(save_interval)

        print_with_rank_check(f"save_interval: {save_interval}", flush=True)

    
    if RESUME_FROM_CHECKPOINT:
        sliding_window_load_path = os.path.join(CFG.resume_checkpoint_path, "sliding_window_standardization.pkl")
        if os.path.isfile(sliding_window_load_path):
            try:
                sliding_window_standardization = SlidingWindowStandardization.load(sliding_window_load_path)
                print_with_rank_check(f"sliding window queue loaded from checkpoint.", flush=True)
            except (IOError, pickle.UnpicklingError) as e:
                print(f"文件加载失败: {e}. init sliding_window_standardization with default value.", flush=True)
                sliding_window_standardization = SlidingWindowStandardization(max_window_size=CFG.welford_max_window_size)
        else:
            print(f"sliding_window_standardization checkpoint not found. Initializing sliding_window_standardization with default value.", flush=True)
            sliding_window_standardization = SlidingWindowStandardization(max_window_size=CFG.welford_max_window_size)
    else:
        sliding_window_standardization = SlidingWindowStandardization(max_window_size=CFG.welford_max_window_size)

    if METRIC_SETTING in {MetricSetting.ELI5}:
        rouge_metric = RougeMetric()
    elif METRIC_SETTING in {MetricSetting.ASQA}:
        if USING_ROUGE_F1_AND_BERTSCORE == True:
            rouge_metric = RougeMetric()
        else:
            judge_model_tokenzier: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(CFG.judge_model_path)


    def update_sliding_window_single_step(
            episode_advantages: torch.Tensor,
            world_size: int,
            device: torch.device,
            dtype: torch.dtype,
            sliding_window: SlidingWindowStandardization
        ):
        """
        单步更新滑动窗口（用于第一个累积周期或边界情况）。
        收集所有rank的episode_advantages并更新滑动窗口。
        """
        all_episode_advantages = torch.empty((world_size, len(episode_advantages)), device=device, dtype=dtype)
        dist.all_gather_into_tensor(all_episode_advantages, episode_advantages)
        all_episode_advantages_lst = [all_episode_advantages[i].cpu().tolist() for i in range(world_size)]
        sliding_window.add_data(all_episode_advantages_lst)

    def update_sliding_window_accumulated(
            accumulated_list: list[torch.Tensor],
            world_size: int,
            device: torch.device,
            dtype: torch.dtype,
            sliding_window: SlidingWindowStandardization
        ):
        """
        累积更新滑动窗口（用于完成一个累积周期的情况）。
        收集所有rank的累积episode_advantages并更新滑动窗口。
        """
        # 构建当前rank的tensor
        episode_advantages_tensor = torch.stack(accumulated_list, dim=0)
        # 收集所有rank的数据
        episode_advantages_all_ranks = torch.empty(
            (world_size,) + episode_advantages_tensor.shape, 
            device=device, 
            dtype=dtype
        )
        dist.all_gather_into_tensor(episode_advantages_all_ranks, episode_advantages_tensor)
        # Order: rank, then step.
        # 格式：[rank0_step0, rank0_step1, ..., rank1_step0, rank1_step1, ...]
        episode_advantages_lst = [
            episode_advantages_all_ranks[i, j].cpu().tolist()
            for i in range(world_size)
            for j in range(len(accumulated_list))
        ]
        sliding_window.add_data(episode_advantages_lst)

    def eval_process():
        """评估的流程，放入闭包函数中方便代码维护"""
        # 初始化
        if rank == 0:
            shared_variables.clean_up()

        eval_start_time = time.time()

        dist.barrier()

        if rank == 0:
            if not USING_SQL:
                t = threading.Thread(
                    target=monitor_and_send,
                    # 不使用 sql
                    args=(
                        shared_variables, CFG.eval_top_k, CFG.llm_reader_name, CFG.prompt_template, clients, INIT_EVAL_FINISH_COUNT_CHECK_INTERVAL, METRIC_SETTING
                    ),
                    daemon=True
                )
            else:
                t = threading.Thread(
                    target=monitor_and_send,
                    # 使用 sql
                    args=(
                        shared_variables, CFG.eval_top_k, CFG.llm_reader_name, CFG.prompt_template, clients, INIT_EVAL_FINISH_COUNT_CHECK_INTERVAL, METRIC_SETTING, 
                        SQL_DB_PARAMS, "test_set_response", self_gpu_type,
                    ),             
                    daemon=True
                )
            t.start()

        dist.barrier()

        if rank == 0:
            inference_worker(ddp_model, test_dataloader, rank, world_size, CFG.eval_chunk_size, SYNC_RERANK_RESULTS_INTERVAL, shared_variables)
        else:
            inference_worker(ddp_model, test_dataloader, rank, world_size, CFG.eval_chunk_size, SYNC_RERANK_RESULTS_INTERVAL)

        # 等线程结束
        if rank == 0:
            t.join()

        dist.barrier()
        print_with_rank_check(f"Evaluation finished with time cost {time.time() - eval_start_time:.3f}s.", flush=True)

        # 获取评估完成的数据
        if rank == 0:
            rerank_results_dic_all = deepcopy(dict(shared_variables.rerank_results_dic))
            reader_results_dic_all = deepcopy(dict(shared_variables.reader_results_dic))

            # 计算最终的评估metric
            if METRIC_SETTING in {MetricSetting.MULTIHOPQA, MetricSetting.MUSIQUE, MetricSetting.AMBIGNQ}:
                new_em, new_f1 = compute_ave_em_f1_from_reader_results(reader_results_dic_all)
                new_em *= 100
                new_f1 *= 100
                eval_metric_results = {
                    'em': new_em,
                    'f1': new_f1,
                }
            elif METRIC_SETTING == MetricSetting.ELI5:
                rouge_L_f1 = compute_ave_rouge_from_reader_results(reader_results_dic_all, return_rouge_L_f1_only=True)
                eval_metric_results = {
                    'rouge-L-f1': rouge_L_f1,
                }
            elif METRIC_SETTING == MetricSetting.ASQA:
                if USING_ROUGE_F1_AND_BERTSCORE == True:
                    rouge_L_f1 = compute_ave_rouge_from_reader_results(reader_results_dic_all, return_rouge_L_f1_only=True)
                    eval_metric_results = {
                        'rouge-L-f1': rouge_L_f1,
                    }
                else:
                    raise NotImplementedError(f"USING_ROUGE_F1_AND_BERTSCORE == False is not inplemented here.")
            else:
                raise NotImplementedError(f"METRIC_SETTING: {METRIC_SETTING} is not inplemented here.")


        # 在源 rank（这里是 0）创建一个列表，里面放要广播的对象；
        # 在其他 rank 里创建一个空容器（例如 [None]）
        if rank == 0:
            # 在 rank=0 上定义任意 Python 对象
            obj_list = [rerank_results_dic_all, reader_results_dic_all, eval_metric_results]
        else:
            # 其他 rank 先放占位符
            obj_list = [None, None, None]

        # 广播这个列表
        dist.broadcast_object_list(obj_list, src=0)

        if rank != 0:
            # 广播之后，在所有 rank 上 obj_list[0] 都是同样的对象
            rerank_results_dic_all, reader_results_dic_all, eval_metric_results = obj_list[0], obj_list[1], obj_list[2]

        return rerank_results_dic_all, reader_results_dic_all, eval_metric_results


    dist.barrier()
    print_with_rank_check(f'start training...', flush=True)
    # 最后的初始化
    ddp_model.train()

    global_step = 0
    if RESUME_FROM_CHECKPOINT:
        global_step = resume_start_global_step

    start_epoch = 0
    if RESUME_FROM_CHECKPOINT:
        start_epoch = resume_start_epoch
    
    episode_advantages_in_accumulation_steps_per_rank_lst = []

    for epoch in range(start_epoch, CFG.training_epochs):
        sampler.set_epoch(epoch)

        steps_to_skip_in_this_epoch = 0
        if RESUME_FROM_CHECKPOINT and epoch == resume_start_epoch:
            steps_to_skip_in_this_epoch = resume_start_step

        desc = f'Epoch {epoch}'
        dataloader_iter = iter(my_dataloader)

        if steps_to_skip_in_this_epoch > 0:
            if rank == 0:
                print(f"Resume: Fast-forwarding dataloader by {steps_to_skip_in_this_epoch} steps using itertools...", flush=True)
            
            # 使用 islice 切片，并用 deque(maxlen=0) 在 C 语言层面高速消耗迭代器
            # 这依然会触发 DataLoader 加载数据，但省去了 Python 循环和数据搬运到 GPU 的开销
            collections.deque(itertools.islice(dataloader_iter, steps_to_skip_in_this_epoch), maxlen=0)

            dist.barrier()
            
            if rank == 0:
                print(f"Resume: Finished skipping {steps_to_skip_in_this_epoch} steps. Synchronization complete. Training starts.", flush=True)

        # start must match the skipped steps, otherwise global_step is wrong.
        data_iterator = enumerate(dataloader_iter, start=steps_to_skip_in_this_epoch)

        progress_bar = tqdm(
            data_iterator,
            total=len(my_dataloader),
            desc=desc,
            initial=steps_to_skip_in_this_epoch 
        )

        for step, batch in progress_bar:
            if ALLOW_REFERENCE_MODEL_UPDATE and not USE_SAVED_REFERENCE_MODEL_METRIC_RESULTS and epoch == 0 and step == 0:
                dist.barrier()

                print_with_rank_check('')

                print_with_rank_check(f'Evaluation started at step {global_step}, epoch {epoch+1} to init reference_eval_results.', flush=True)

                ddp_model.eval()
                dist.barrier()

                rerank_results_dic_all, reader_results_dic_all, reference_eval_results = eval_process()
                
                ddp_model.train()
                dist.barrier()

                for k in reference_eval_results.keys():
                    print_with_rank_check(f"Metric {k}: INIT: {reference_eval_results[k]}")

                if rank == 0:
                    # 保存结果为文件
                    results_savepath = set_save_path(epoch, global_step)
                    save_eval_results(
                        rerank_results_dic_all, 
                        reader_results_dic_all, 
                        reference_eval_results, 
                        results_savepath, 
                        epoch, 
                        global_step
                    )

                dist.barrier()

            print_with_rank_check('')

            query = batch['query']
            passages = batch['passages']
            if MODEL_SETTING not in {"qwen3_reranker"}:
                model_inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
            else:
                pairs = [ddp_model.module.format_instruction(query, passage) for passage in passages]
                model_inputs = ddp_model.module.process_inputs(pairs)
            batch_len = model_inputs['input_ids'].shape[0]

            print_with_rank_check(f"case_id: {batch['case_id']}")
            if METRIC_SETTING in {MetricSetting.MULTIHOPQA, MetricSetting.MUSIQUE}:
                print_with_rank_check(f'query: {batch["query"]}')
                print_with_rank_check(f'answer: {batch["answer"]}', flush=True)
            elif METRIC_SETTING == MetricSetting.ELI5:
                print_with_rank_check(f'query_title: {batch["query_original"]["title"]}')
                print_with_rank_check(f'query: {batch["query_original"]["selftext"]}')
            elif METRIC_SETTING in {MetricSetting.AMBIGNQ, MetricSetting.ASQA}:
                print_with_rank_check(f'query: {batch["query"]}')
                print_with_rank_check(f'answers: {batch["answers"]}', flush=True)
            else:
                raise NotImplementedError(f"METRIC_SETTING: {METRIC_SETTING} is not supported.")


            whole_step_start_time = time.time()


            all_loss = torch.zeros(1, dtype=self_dtype, device=self_device)

            step_seed = seed_generator.generate_seed()

            set_seed(step_seed)
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=CFG.use_amp):
                with torch.no_grad():
                    if MODEL_SETTING not in {"qwen3_reranker"}:
                        all_logits_nograd = compute_single_case_logits(ddp_model, model_inputs, CFG.train_chunk_size)
                    else:
                        all_logits_nograd = ddp_model.module.compute_single_case_logits(model_inputs, CFG.train_chunk_size)

            print_with_rank_check(f"all_logits_nograd: {all_logits_nograd}", flush=True)

            if not USE_SAVED_REFERENCE_MODEL_RERANK_RESULTS:
                with torch.no_grad():
                    if MODEL_SETTING not in {"qwen3_reranker"}:
                        all_logits_reference = compute_single_case_logits(reference_model, model_inputs, batch_len, self_device)
                    else:
                        with torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=CFG.use_amp):
                            all_logits_reference = reference_model.compute_single_case_logits(model_inputs, CFG.eval_chunk_size)
            else:
                all_logits_reference_dic = reference_logits_data[batch['case_id']]
                all_logits_reference = torch.tensor([all_logits_reference_dic[passage_id] for passage_id in batch['passage_ids']], device=self_device, dtype=self_dtype)

            """计算状态价值函数V(s)"""
            if USE_SAVED_REFERENCE_MODEL_READER_RESULTS:
                reranker_per_action_responses = get_episode_v_responses_from_cache(reader_results_bm25_reranker_raw, batch['case_id'], CFG.train_top_k)
            else:
                if not USING_SQL:
                    # 不使用 sql
                    reranker_per_action_responses = get_episode_v_responses_with_rank_check(
                        all_logits_reference, query, passages, CFG.llm_reader_name, CFG.prompt_template, rank, world_size, CFG.train_top_k,
                    )
                else:
                    # 使用sql
                    with conn.cursor() as cur:
                        reranker_per_action_responses = get_episode_v_responses_with_rank_check(
                            all_logits_reference=all_logits_reference,
                            query=query,
                            passages=passages,
                            case_id=batch['case_id'],
                            passage_ids=batch['passage_ids'],
                            conn=conn,
                            cur=cur,
                            llm_reader_name=CFG.llm_reader_name,
                            prompt_template=CFG.prompt_template,
                            rank=rank,
                            world_size=world_size,
                            gpu_type=self_gpu_type,
                            top_k=CFG.train_top_k,
                        )

            if METRIC_SETTING != MetricSetting.ELI5:
                print_with_rank_check(f'reranker_episode_v_responses: {reranker_per_action_responses}')
            
            """然后计算价值函数作为基线"""
            if METRIC_SETTING in {MetricSetting.MULTIHOPQA, MetricSetting.MUSIQUE}:
                episode_v_values = compute_reward(reranker_per_action_responses, [batch['answer']] * CFG.train_top_k, f1_lambda=CFG.f1_lambda, hit_lambda=CFG.hit_lambda, device=self_device, dtype=self_dtype)
            elif METRIC_SETTING == MetricSetting.ELI5:
                rouge_L_f1_v_values = rouge_metric.compute_scores(reranker_per_action_responses, [batch['answer']] * CFG.train_top_k, return_rouge_L_f1_only=True, avg=False)
                episode_v_values = torch.tensor(rouge_L_f1_v_values, device=self_device, dtype=self_dtype)
            elif METRIC_SETTING == MetricSetting.AMBIGNQ:
                episode_v_values = compute_episode_rewards_for_multi_answers(reranker_per_action_responses, batch['answers'], f1_lambda=CFG.f1_lambda, hit_lambda=CFG.hit_lambda, device=self_device, dtype=self_dtype)
            elif METRIC_SETTING == MetricSetting.ASQA:
                if USING_ROUGE_F1_AND_BERTSCORE == True:
                    # ASQA training examples have a single answer.
                    rouge_L_f1_v_values_response = rouge_metric.compute_scores(reranker_per_action_responses, batch['answers'] * CFG.train_top_k, return_rouge_L_f1_only=True, avg=False)
                    rouge_L_f1_v_values = torch.tensor(rouge_L_f1_v_values_response, device=self_device, dtype=self_dtype)
                    bertscore_v_values_response = send_to_compute_bertscores(reranker_per_action_responses, batch['answers'] * CFG.train_top_k, CFG.bertscore_server_url)
                    bertscore_v_values = torch.tensor(bertscore_v_values_response["scores"], device=self_device, dtype=self_dtype)
                    print_with_rank_check(f"rouge_L_f1_v_values: {rouge_L_f1_v_values}", flush=True)
                    print_with_rank_check(f"bertscore_v_values: {bertscore_v_values}", flush=True)
                    episode_v_values = rouge_L_f1_v_values + bertscore_v_values
                else:
                    judge_model_scores = send_to_judge_model_to_get_scores(
                        reranker_per_action_responses,
                        batch['answers'] * CFG.train_top_k,
                        batch['query'],
                        CFG.judge_model_server_urls,
                        world_size=world_size,
                        rank=rank,
                        judge_model_tokenizer=judge_model_tokenzier,
                        judge_model_name=CFG.judge_model_name,
                    )
                    episode_v_values = torch.tensor(judge_model_scores, device=self_device, dtype=self_dtype)
            else:
                raise NotImplementedError(f"METRIC_SETTING: {METRIC_SETTING} is not supported.")

            print_with_rank_check(f'episode_v_values: {episode_v_values}', flush=True)



            """获取不计梯度的最大概率动作(top-k)，以及计算"策略网络"中该动作对应的对数概率"""
            action_nograd, log_prob_nograd, per_action_probs_nograd = select_max_prob_action(
                all_logits_nograd, 
                CFG.train_top_k, 
                return_probs=True, 
                device=self_device
            )
            # only for print. not join the computation
            per_action_log_probs_nograd = torch.log(per_action_probs_nograd)

            print_with_rank_check(f'action:', action_nograd)
            print_with_rank_check(f'action_probs:', per_action_probs_nograd)
            print_with_rank_check(f'per_action_log_probs_nograd:', per_action_log_probs_nograd)

            """计算该动作在参考模型中对应的概率"""
            log_prob_reference, per_action_probs_reference = get_action_log_prob(
                all_logits_reference, 
                action_nograd, 
                return_probs=True, 
                device=self_device
            )
            per_action_log_probs_reference = torch.log(per_action_probs_reference)

            print_with_rank_check(f"per_action_probs_reference: {per_action_probs_reference}", flush=True)
            print_with_rank_check(f"per_action_log_probs_reference: {per_action_log_probs_reference}", flush=True)



            """获取action对应的段落， 然后和query一起送入reader模型，得到模型输出的结果"""
            action_passages = [passages[action_nograd[i]] for i in range(len(action_nograd))]

            if not USING_SQL:
                # 不使用 sql
                episode_action_responses = get_episode_reader_response_with_rank_check(
                    query,
                    action_passages,
                    CFG.llm_reader_name, 
                    CFG.prompt_template, 
                    clients,
                    rank, world_size,
                    METRIC_SETTING,
                )
            else:
                # 使用 sql
                with conn.cursor() as cur:
                    episode_action_responses = get_episode_reader_response_with_rank_check(
                        query=query,
                        action_passages=action_passages,
                        case_id=batch['case_id'],
                        action_passage_ids=[batch['passage_ids'][action_nograd[i]] for i in range(len(action_nograd))],
                        conn=conn,
                        cur=cur,
                        case_type="train_action_response",
                        gpu_type=self_gpu_type,
                        llm_reader_name=CFG.llm_reader_name,
                        prompt_template=CFG.prompt_template,
                        clients=clients,
                        rank=rank,
                        world_size=world_size,
                        metric_setting=METRIC_SETTING,
                    )
            if METRIC_SETTING != MetricSetting.ELI5:
                print_with_rank_check(f"episode_responses: {episode_action_responses}")

            """计算episode_rewards"""
            """避免可能出现的bug: episode_action_responses 中可能因为一些原因有None"""
            try:
                if METRIC_SETTING in {MetricSetting.MULTIHOPQA, MetricSetting.MUSIQUE}:
                    episode_rewards = compute_reward(episode_action_responses, [batch['answer']] * CFG.train_top_k, f1_lambda=CFG.f1_lambda, hit_lambda=CFG.hit_lambda, device=self_device, dtype=self_dtype)
                elif METRIC_SETTING == MetricSetting.ELI5:
                    rouge_L_f1_rewards = rouge_metric.compute_scores(episode_action_responses, [batch['answer']] * CFG.train_top_k, return_rouge_L_f1_only=True, avg=False)
                    episode_rewards = torch.tensor(rouge_L_f1_rewards, device=self_device, dtype=self_dtype)
                elif METRIC_SETTING == MetricSetting.AMBIGNQ:
                    episode_rewards = compute_episode_rewards_for_multi_answers(episode_action_responses, batch['answers'], f1_lambda=CFG.f1_lambda, hit_lambda=CFG.hit_lambda, device=self_device, dtype=self_dtype)
                elif METRIC_SETTING == MetricSetting.ASQA:
                    if USING_ROUGE_F1_AND_BERTSCORE == True:
                        rouge_L_f1_rewards_response = rouge_metric.compute_scores(episode_action_responses, batch['answers'] * CFG.train_top_k, return_rouge_L_f1_only=True, avg=False)
                        rouge_L_f1_rewards = torch.tensor(rouge_L_f1_rewards_response, device=self_device, dtype=self_dtype)
                        bertscore_rewards_response = send_to_compute_bertscores(reranker_per_action_responses, batch['answers'] * CFG.train_top_k, CFG.bertscore_server_url)
                        bertscore_rewards = torch.tensor(bertscore_rewards_response["scores"], device=self_device, dtype=self_dtype)
                        print_with_rank_check(f"rouge_L_f1_rewards: {rouge_L_f1_rewards}", flush=True)
                        print_with_rank_check(f"bertscore_rewards: {bertscore_rewards}", flush=True)
                        episode_rewards = rouge_L_f1_rewards + bertscore_rewards
                    else:
                        judge_model_scores = send_to_judge_model_to_get_scores(
                            episode_action_responses,
                            batch['answers'] * CFG.train_top_k,
                            query,
                            CFG.judge_model_server_urls,
                            world_size=world_size,
                            rank=rank,
                            judge_model_tokenizer=judge_model_tokenzier,
                            judge_model_name=CFG.judge_model_name,
                        )
                        episode_rewards = torch.tensor(judge_model_scores, device=self_device, dtype=self_dtype)
                else:
                    raise NotImplementedError(f"METRIC_SETTING: {METRIC_SETTING} is not supported.")

            except Exception as e:
                # print_with_rank_check(f"[Error] in compute_reward: {e}", flush=True)

                # 1. 获取完整的 traceback 字符串
                tb_str = traceback.format_exc()
                
                # 2. 准备包含上下文的详细错误消息
                error_message = (
                    f"==================== [ERROR] in compute_reward ====================\n"
                    f"METRIC_SETTING: {METRIC_SETTING}\n"
                    f"Exception Type: {type(e).__name__}\n"
                    f"Exception Details: {e}\n"
                    f"----------------------- Traceback -----------------------\n"
                    f"{tb_str}\n"
                    f"================================================================="
                )
                
                # 3. 打印完整的错误信息
                print_with_rank_check(error_message, flush=True)

                # 这里的处理方式是直接复制状态函数
                print_with_rank_check(f"[WARNING] Can't compute reward, copy state value instead.", flush=True)
                episode_rewards = episode_v_values.clone().detach()

            print_with_rank_check(f"episode_rewards: {episode_rewards}", flush=True)


            """计算广义优势估计"""
            episode_advantages = compute_GAE(episode_rewards, episode_v_values, device=self_device, dtype=self_dtype)

            print_with_rank_check(f"episode_advantages: {episode_advantages}", flush=True)

            
            # 判断是否是第一个累积周期（用于快速初始化统计量）
            is_first_accumulation_cycle = (
                (not RESUME_FROM_CHECKPOINT and epoch == 0 and (step + 1) <= ACCUMULATION_STEPS_PER_RANK) or
                (RESUME_FROM_CHECKPOINT and epoch == resume_start_epoch and step <= resume_start_step + ACCUMULATION_STEPS_PER_RANK)
            )
            
            # 判断是否是累积步数的边界
            is_accumulation_boundary = (
                (step + 1) % ACCUMULATION_STEPS_PER_RANK == 0 or 
                (step + 1) == len(my_dataloader)
            )
            
            # 判断是否是累积周期的第一步
            is_accumulation_cycle_start = (step + 1) % ACCUMULATION_STEPS_PER_RANK == 1
            
            # 判断是否是边界且列表为空（特殊情况：epoch结束且刚好是第一步）
            is_boundary_with_empty_list = is_accumulation_boundary and len(episode_advantages_in_accumulation_steps_per_rank_lst) == 0
            
            if is_first_accumulation_cycle or is_boundary_with_empty_list:
                # 情况1 & 2: 单步更新（第一个累积周期 或 边界且列表为空）
                update_sliding_window_single_step(
                    episode_advantages, world_size, self_device, self_dtype, sliding_window_standardization
                )
            elif is_accumulation_cycle_start and not is_accumulation_boundary:
                # 情况3: 累积周期的第一步（且不是边界）
                episode_advantages_in_accumulation_steps_per_rank_lst.clear()
                episode_advantages_in_accumulation_steps_per_rank_lst.append(episode_advantages)
            elif is_accumulation_boundary:
                # 情况4: 累积周期的最后一步
                episode_advantages_in_accumulation_steps_per_rank_lst.append(episode_advantages)
                update_sliding_window_accumulated(
                    episode_advantages_in_accumulation_steps_per_rank_lst, 
                    world_size, self_device, self_dtype, sliding_window_standardization
                )
                episode_advantages_in_accumulation_steps_per_rank_lst.clear()
            else:
                # 情况5: 累积周期的中间步骤
                episode_advantages_in_accumulation_steps_per_rank_lst.append(episode_advantages)

            # 对本 episode 的优势值进行标准化
            # 计算均值和标准差
            current_mean, current_std = sliding_window_standardization.get_mean(), sliding_window_standardization.get_std()
            # 注意：如果 current_std 很小，需要避免除零
            epsilon = 1e-8
            if current_std < epsilon:
                episode_advantages_standardized = torch.zeros_like(episode_advantages, device=self_device, dtype=self_dtype)
            else:
                episode_advantages_standardized = (episode_advantages - current_mean) / (current_std + epsilon)

            print_with_rank_check(f"current_mean: {current_mean}, current_std: {current_std}")
            print_with_rank_check(f"episode_advantages_standardized: {episode_advantages_standardized}", flush=True)





            """在计算梯度下，按批重新计算所有的 logits 并梯度累积，以供训练"""
            set_seed(step_seed)
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=CFG.use_amp):

                forward_steps_count = math.ceil(batch_len / CFG.train_chunk_size)

                for forward_step in range(forward_steps_count):
                    start_chunk_idx = forward_step * CFG.train_chunk_size
                    end_chunk_idx = min((forward_step + 1) * CFG.train_chunk_size, batch_len)

                    model_inputs_chunk = {k: v[start_chunk_idx:end_chunk_idx] for k, v in model_inputs.items()}
                    # 如果只有一个数据，改变一下形状
                    if model_inputs_chunk['input_ids'].shape[0] == 1:
                        for k, v in model_inputs_chunk.items():
                            model_inputs_chunk[k] = v.reshape(1, -1)

                    def latter_process():
                        nonlocal all_loss

                        # 前向传播
                        if MODEL_SETTING not in {"qwen3_reranker"}:
                            chunk_logits = compute_logits(ddp_model, model_inputs_chunk)
                        else:
                            chunk_logits = ddp_model.module.compute_logits(model_inputs_chunk)
                    
                        """拼接logits, 并再计算动作和对数概率"""
                        logits_for_computing = torch.cat([all_logits_nograd[:start_chunk_idx], chunk_logits, all_logits_nograd[end_chunk_idx:]], dim=0)

                        action, log_prob, per_action_probs = select_max_prob_action(logits_for_computing, CFG.train_top_k, return_probs=True, device=self_device)
                        per_action_log_probs = torch.log(per_action_probs)

                        """计算PPO的目标函数"""
                        # 计算该动作在两个策略模型中的比率ratio
                        # episode_ratios = per_action_probs / per_action_probs_reference
                        # 用对数概率计算，避免数值稳定性问题
                        episode_ratios = torch.exp(per_action_log_probs - per_action_log_probs_reference)

                        # PPO clip on standardized advantages, plus a KL penalty.
                        # 计算PPO clip
                        surr1 = episode_ratios * episode_advantages_standardized
                        surr2 = torch.clamp(episode_ratios, 1 - CFG.clip_epsilon, 1 + CFG.clip_epsilon) * episode_advantages_standardized
                        # 这是“要最大化”的目标，所以这里用负号来做“要最小化”的 loss
                        chunk_ppo_loss = - torch.min(surr1, surr2).mean()

                        # 计算KL散度
                        episode_kl_div = compute_kl_div_estimation(p=per_action_probs, q=per_action_probs_reference)
                        chunk_kl_loss = CFG.kl_loss_beta * episode_kl_div.mean()

                        # 总的policy损失 = PPO_loss + KL惩罚
                        chunk_loss = chunk_ppo_loss + chunk_kl_loss


                        chunk_loss /= CFG.K_conceptual
                        all_loss += chunk_loss.clone().detach()

                        chunk_loss /= ACCUMULATION_STEPS_PER_RANK

                        # 反向传播
                        scaler.scale(chunk_loss).backward()


                    # no_sync must wrap the forward pass; DDP registers sync hooks during forward.
                    if ((step + 1) % ACCUMULATION_STEPS_PER_RANK == 0 or (step + 1) == len(my_dataloader)) \
                        and forward_step == forward_steps_count - 1:

                        latter_process()

                        time_str = print_time(return_str_instead=True)
                        print_with_rank_check(time_str + f", global_step: {global_step}, epoch: {epoch}, step: {step}", flush=True)

                    else:
                        with ddp_model.no_sync():

                            latter_process()

            if ((step + 1) % ACCUMULATION_STEPS_PER_RANK == 0 or (step + 1) == len(my_dataloader)):
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
                scheduler.step()

            global_step += world_size

            whole_step_end_time = time.time()

            print_with_rank_check(f"computing all loss time cost: {whole_step_end_time - whole_step_start_time:.3f}s", flush=True)

            print_with_rank_check(f"loss: {all_loss}", flush=True)

            lr = optimizer.param_groups[0]['lr']

            print_with_rank_check(f"learning_rate: {lr}", flush=True)

            if ALLOW_REFERENCE_MODEL_UPDATE:
                if global_step >= check_for_reference_model_interval:
                    # 在进行评估之前，确保所有进程都完成了训练步骤
                    dist.barrier()

                    check_for_reference_model_interval += CFG.check_for_reference_model_steps
                    
                    print_with_rank_check(f"Evaluation started at step {global_step}, epoch {epoch+1}.", flush=True)

                    ddp_model.eval()
                    dist.barrier()

                    rerank_results_dic_all, reader_results_dic_all, new_eval_results = eval_process()
                
                    ddp_model.train()
                    dist.barrier()

                    print_with_rank_check(f'Evaluation results and comparation at {epoch+1}epoch, {global_step}step:', flush=True)
                    for k in reference_eval_results.keys():
                        print_with_rank_check(f"Metric {k}: OLD: {reference_eval_results[k]} -> NEW: {new_eval_results[k]}")

                    dist.barrier()

                    if rank == 0:
                        # 保存模型
                        if MODEL_SETTING in {'gte', 'jina'}:
                            model_savepath = save_special_model(ddp_model, CFG.additional_file_path, CFG.additional_file_names, epoch, global_step, return_savepath=True)
                        else:
                            model_savepath = save_model(ddp_model, epoch, global_step, return_savepath=True)

                        # 保存结果为文件
                        save_eval_results(
                            rerank_results_dic_all, 
                            reader_results_dic_all, 
                            new_eval_results, 
                            model_savepath, 
                            epoch, 
                            global_step
                        )

                        save_checkpoint(epoch, global_step, optimizer, scheduler, scaler, model_savepath)
                        sliding_window_save_path = os.path.join(model_savepath, "sliding_window_standardization.pkl")
                        sliding_window_standardization.save(sliding_window_save_path)
                        print_with_rank_check(f'Rank 0: sliding_window_standardization status saved.', flush=True)

                    dist.barrier()

                    # Update the reference model unless any metric drops. Equal metrics still update.
                    UPDATE_REFERENCE_NEEDED = True
                    for k in new_eval_results.keys():
                        if reference_eval_results[k] > new_eval_results[k]:
                            UPDATE_REFERENCE_NEEDED = False
                            break

                    if UPDATE_REFERENCE_NEEDED:

                        if USE_SAVED_REFERENCE_MODEL_RERANK_RESULTS:
                            USE_SAVED_REFERENCE_MODEL_RERANK_RESULTS = False
                        if USE_SAVED_REFERENCE_MODEL_READER_RESULTS:
                            USE_SAVED_REFERENCE_MODEL_READER_RESULTS = False

                        print_with_rank_check(f"Results improvement is observed at {epoch+1}epoch, {global_step}step. UPDATE reference model...")

                        reference_eval_results = deepcopy(new_eval_results)

                        dist.barrier()
                        unwrapped_model = ddp_model.module
                        reference_model.model.load_state_dict(unwrapped_model.model.state_dict())
                        print(f"Reference model updated.", flush=True)

                    else:
                        print_with_rank_check(f"Results improvement is NOT observed at {epoch+1}epoch, {global_step}step.")

                    # 避免rank0保存模型，其他rank已经接着训练的情况
                    dist.barrier()

            else:
                if global_step >= save_interval:
                    save_interval += CFG.save_steps
                    # 在保存之前，确保所有进程都完成了训练步骤
                    dist.barrier()
                    # 只有 rank 0 的进程负责保存模型
                    if rank == 0:
                        if MODEL_SETTING in {'gte', 'jina'}:
                            model_savepath = save_special_model(ddp_model, CFG.additional_file_path, CFG.additional_file_names, epoch, global_step, return_savepath=True)
                        else:
                            model_savepath = save_model(ddp_model, epoch, global_step, return_savepath=True)
                        save_checkpoint(epoch, global_step, optimizer, scheduler, scaler, model_savepath)
                        sliding_window_save_path = os.path.join(model_savepath, "sliding_window_standardization.pkl")
                        sliding_window_standardization.save(sliding_window_save_path)
                        print_with_rank_check(f'Rank 0: sliding_window_standardization status saved.', flush=True)

                    # 避免rank0保存模型，其他rank已经接着训练的情况
                    dist.barrier()


        dist.barrier()
        if rank == 0:
            if MODEL_SETTING in {'gte', 'jina'}:
                model_savepath = save_special_model(ddp_model, CFG.additional_file_path, CFG.additional_file_names, epoch, return_savepath=True)
            else:
                model_savepath = save_model(ddp_model, epoch, return_savepath=True)
            save_checkpoint(epoch, global_step, optimizer, scheduler, scaler, model_savepath)
            sliding_window_save_path = os.path.join(model_savepath, "sliding_window_standardization.pkl")
            sliding_window_standardization.save(sliding_window_save_path)
            print_with_rank_check(f'Rank 0: sliding_window_standardization status saved.', flush=True)

        # 避免rank0保存模型，其他rank已经接着训练的情况
        dist.barrier()






def print_with_rank_check(*args, **kwargs):
    print(*args, **kwargs)


def main():
    print(f"train_chunk_size: {CFG.train_chunk_size}", flush=True)
    print(f"prompt template: {CFG.prompt_template.__class__.__name__}", flush=True)
    print(f'please CHECK your config file: {CFG.__class__.__name__}', flush=True)
    print_time()

    if USING_SQL:
        global conn
        print("正在连接到 PostgreSQL 数据库...")
        conn = psycopg2.connect(**SQL_DB_PARAMS)

        with conn.cursor() as cur:
            cur.execute('SELECT version()')
            db_version = cur.fetchone()
            print(f"成功连接！PostgreSQL 数据库, 版本：{db_version[0]}")

    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print_with_rank_check(f"rank: {rank}, local_rank: {local_rank}, world_size: {world_size}", flush=True)
    if torch.cuda.is_available():
        print_with_rank_check("CUDA is available. Device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print_with_rank_check(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print_with_rank_check("No GPU available.")


    redirect_output(rank, CFG.DDP_log_dir, CFG.DDP_log_filename_prefix)

    timeout_delta = datetime.timedelta(minutes=120)
    dist.init_process_group(
        backend='nccl', 
        init_method="env://",
        timeout=timeout_delta, 
    )

    # 设置当前进程的设备。如果一个进程有多个GPU，设置其管理的第一个GPU为主设备。
    main_device = local_rank * GPUS_PER_RANK

    # 不设置不会影响dist.all_gather(), dist.all_reduce()操作，但会导致dist.all_gather_object()卡死。
    # 原因是不设置的话，每个rank的主设备都是0号GPU，即torch.cuda.current_device() = 0
    torch.cuda.set_device(main_device)

    train(rank, local_rank, world_size)


    if USING_SQL:
        # global variable conn
        if conn:
            conn.close()
            print("数据库连接已关闭。")


    # 训练结束，清理分布式环境
    dist.destroy_process_group()



if __name__ == "__main__":

    main()