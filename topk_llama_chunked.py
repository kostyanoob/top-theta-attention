import itertools
import pdb
from typing import Tuple
import numpy as np
import torch
from transformers.models.llama.modeling_llama import *
import torch.nn.functional as F
import math

def memory_efficient_sum_exp_attn_minus_max(attn: torch.Tensor,
                                            max_attn_per_row: torch.Tensor,
                                            dtype: torch.dtype):
    """
    for every batch, head, attention_row compute the sum of: exponentiated attention row minus row's max.
    
    output_tensor[b,h,row] = sum( exp(attn[b,h,row,:] - max_attn_per_row(exp(attn[b,h,row,:])) )

    Args:
        attn: float tensor of shape [B, N, q_chunk_len, kv_seq_len] - pre-softmax attention scores
        max_attn_per_row: float tensor of shape [B, N, q_chunk_len] - per-attention-row maximum values of the pre-softmax-attension scores
        dtype: numeric data type in which the exponentiation should be taking place: recommended: torch.float32

    Return:
        tensor of shape [B, N, q_chunk_len] - with the sums of exponentiated 
        rows of <attn> (with maxiums subtracted).
    """
    B, N, q_chunk_len, kv_seq_len = attn.shape
    if q_chunk_len * kv_seq_len < 200 * 42_000:  # arbitrary limit based on llama3.1-8B on 48GB GPU
        # one-shot torch operation
        output_tensor = memory_efficient_sum_reduction(torch.exp(attn.to(dtype=dtype) - max_attn_per_row.unsqueeze(-1).to(dtype=dtype)), dim=-1)
    else:
        # iterative chunked reduction (chunk again across the q_len dimension)
        q_chunk_secondary_len = max(int(q_chunk_len / 10), 1)
        output_tensor = torch.zeros(B, N, q_chunk_len, dtype=dtype, device=attn.device)
        for start_idx in range(0, q_chunk_len, q_chunk_secondary_len):
            end_idx = min(start_idx + q_chunk_secondary_len, q_chunk_len)
            output_tensor[:,:,start_idx:end_idx] = memory_efficient_sum_reduction(torch.exp(attn[:,:,start_idx:end_idx,:].to(dtype=dtype) - max_attn_per_row[:,:,start_idx:end_idx].unsqueeze(-1).to(dtype=dtype)), dim=-1)
    return output_tensor


def memory_efficient_sum_reduction(input_tensor:torch.Tensor, dim:Tuple) -> torch.Tensor:
    """
    reduce the input tensor with shape (B, N, q_chunk_len, kv_seq_len) across the dimensions <dim>
    return tensor of interger counters:
      if dim = (2,3) return tensor of shape (B, N)
      if dim = (2) return tensor of shape (B, N, kv_seq_len)
      if dim = (3) return tensor of shape (B, N, q_chunk_len)
    """
    if len(input_tensor.shape) != 4:
        raise ValueError(f"memory_efficient_sum_reduction() supports only input tensor with shape of length 4.")

    B, N, q_chunk_len, kv_seq_len = input_tensor.shape
    
    if q_chunk_len * kv_seq_len < 200 * 42_000:  # arbitrary limit based on llama3.1-8B on 48GB GPU
        # one-shot torch reduction
        output_tensor = input_tensor.sum(dim=dim)
    elif dim in [(2, 3), (-2, -1)]:
        # iterative chunked reduction (chunk again across the q_len dimension)
        q_chunk_secondary_len = max(int(q_chunk_len / 10), 1)
        output_tensor = torch.zeros(B, N, device=input_tensor.device)
        for start_idx in range(0, q_chunk_len, q_chunk_secondary_len):
            end_idx = min(start_idx + q_chunk_secondary_len, q_chunk_len)
            output_tensor += input_tensor[:, :, start_idx:end_idx, :].sum(dim=dim)
    elif dim in [2, -2, (2,), (-2,)]:
        # iterative chunked reduction (chunk again across the q_len dimension)   (B, N, q_chunk_len, kv_seq_len) --> (B, N, kv_seq_len)
        q_chunk_secondary_len = max(int(q_chunk_len / 10), 1)
        output_tensor = torch.zeros(B, N, kv_seq_len, device=input_tensor.device)
        for start_idx in range(0, q_chunk_len, q_chunk_secondary_len):
            end_idx = min(start_idx + q_chunk_secondary_len, q_chunk_len)
            output_tensor += input_tensor[:, :, start_idx:end_idx, :].sum(dim=dim)            
    elif dim in [3, -1, (3,), (-1,),]:
        # iterative chunked reduction (chunk again across the q_len dimension)  (B, N, q_chunk_len, kv_seq_len) --> (B, N, q_chunk_len)
        q_chunk_secondary_len = max(int(q_chunk_len / 10), 1)
        output_tensor = torch.zeros(B, N, q_chunk_len, device=input_tensor.device)
        for start_idx in range(0, q_chunk_len, q_chunk_secondary_len):
            end_idx = min(start_idx + q_chunk_secondary_len, q_chunk_len)
            output_tensor[:, :, start_idx:end_idx] = input_tensor[:, :, start_idx:end_idx, :].sum(dim=dim)
    else:
        raise ValueError(f"Unsupported dimesnsion argument for memory_efficient_sum_reduction. Use either dim=(2,3), or dim=2, or dim=3.")

    return output_tensor   

def lower_triangular_slice(start_row_idx, end_row_idx, num_cols, device) -> torch.Tensor:
    """
    Materialize a horizontal slice of a lower triangular boolean mask.

    Args:
        start_row_idx (int): Starting row index of the slice
        end_row_idx (int): Last number of row to include in putput slice (exclusive)
        num_cols (int): Size of the full square matrix (n x n).

    Returns:
        torch.Tensor: A boolean tensor of shape (chunk_max_size, num_cols) representing the slice.
    """
    # Create a range for the row indices of the slice
    row_indices = torch.arange(start_row_idx, min(end_row_idx , num_cols)).unsqueeze(1)  # Shape: (slice_size, 1)
    col_indices = torch.arange(num_cols).unsqueeze(0)  # Shape: (1, num_cols)
    # Generate the lower triangular mask for the slice
    mask_slice = row_indices >= col_indices  # Broadcasting to compare row and column indices
    return mask_slice.to(device)

def aggregate_threshold_list(row_th_lst:List[float], calib_add_sigma=0.0) -> float:
    """
    Aggregating a list of thresholds into a single scalar

    gets a list row_th_lst=[t1, t2, ...., tc] and a factor calib_add_sigma
    returns mean(v) + calib_add_sigma * std(v)
    """
    if len(row_th_lst) == 1:
        return row_th_lst[0]
    else:
        v = np.array(row_th_lst)
        return v.mean() + calib_add_sigma * v.std()

class TopK_LLamaAttention(LlamaAttention):
    def __init__(self, config : LlamaConfig, layer_idx: int, reduce_gpu_mem: bool = False, products_dir_path: str = "products"):
        try:
            super().__init__(config, layer_idx)
        except:
            super().__init__(config)
        self.to(config.torch_dtype) # Note in theory the super class should instantiate the module in torch_dtype. Might be obsolote for newer versions of transformers
        self.max_q_chunk_size = 1536  # for chunked prefill - this will be the longest seq-len dimension processed at a time while computing Softmax(QK^T)V
        self.K = -1            # K value
        self.id = layer_idx           # Layer id
        self.calibrate = False # Enable calibration mode (could be turned off by the obect itself once it processes the desired number of calibration samples)
        self.calibration_phase = False # general flag that marks the calibration mode (cannot be turned off by the object itself - helps identifying generative decoding of the last calibration sample still belongs to calibration, when the self.calibrate=False hence no need to dump its products). Essentially, when calibration_phase=True amd calibrate=False it means that the calibration phase is finalizing, no more calibration takes place - but the current input is still from the calibration set.
        self.mode = 3          # 0-TH, 1-TOPK, other than 0/1 for baseline
        self.placement = 'none' # 'pre-sofmtax' or 'post-softmax' - for topk/th; 'none' - for baseline

        self.num_calib_requests=0  # total number of requests to use for calibration
        self.obt_calib_requests=0  # current number of processed calibration requests      
        self.calib_load_path=""    # path to load the thresholds or sdc values instead of calibrating from scratch
        self.calib_tac=False       # topk-at-calibration (applies only for top-th)
        self.calib_add_sigma=0.0   # add this many standard deviations to the average threshold, when aggregating per-calib-sample thresholds at the end of calibration.
        self.calib_sample_frac=0.1 # fraction of the attention rows to actuall ues for calibration
        self.th_list=[]            # final th list vs seqlen {head_num: {LEN : TH}}
        self.th_num_samples=[]     # num samples for every seq len in calibration {head_num: {LEN : num_samples_LEN}}
        self.th_fit_params=None    # not used
        self.test_layer = None     # Layer to be tested, None -> all layers tested

        self.rng = np.random.default_rng(42)
        self.reduce_gpu_mem = reduce_gpu_mem
        self.products_dir_path = products_dir_path  # per-layer thresholds from the calibration are written here
        self.dump_qkv = False      # dump per-layer qKV matrices. Careful: they don't distinct different samples, if the samples will have same sequence length - they will be overwritten by hte latest one having the same seqlen
        self.dump_stats_set = set() # set of stats to dump
        self.sdc = 'none'           # sdc = softmax denominatr compensation
        self.sdc_scale = 0.0        # coefficient that mul;iplies the sdc term
        self.sdc_list = []          # sdc terms per calibration sample (for sdc='offline-calibrated' only)
        self.sdc_num_samples = []   # number of calibration samples obtained (for sdc='offline-calibrated' only)

        self.vmc = False            # v-mean compensation (applies only for top-k/th)
        self.capk = False           # cap the number of row elements that survive the thresholding (mode=0 only) 
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        # padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        kv_seq_len = key_states.shape[-2]

        # Grouped-Query-Attention (GQA)
        # Repeat the k, v states according to the number of groups (so that dimensions of k, v will *match* the one of Q)
        # This will have no effect when self.num_key_value_groups == 1 (i.e. no grouped query attention is applied)
        # Warning: self.num_key_value_groups is a misleading name. It is beter to be called kv_group_size or num_q_heads_per_kv_head
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # k = [bsz, num_heads=num_key_value_groups*num_key_value_heads, q_len, head_dim]      
        # v = [bsz, num_heads=num_key_value_groups*num_key_value_heads, q_len, head_dim]
        
        # Initialize attention outputs
        attn_output = torch.zeros(
            (bsz, self.num_heads, q_len, self.head_dim), dtype=query_states.dtype, device=query_states.device
        )
        attn_weights = None if not output_attentions else [] # TODO change this list append to appending across the q_len dimension
        inference_phase = "generative_decoding" if q_len==1 else "prefill"

        # Process in chunks along q_len - each time process "q_chunk_len" rows of attention, and output
        if kv_seq_len < 32000:
            max_q_chunk_size_effective = self.max_q_chunk_size
        elif 32_000 <= kv_seq_len < 42_000:
            max_q_chunk_size_effective = 800 if self.mode not in [0,1] else 400 # radically reduce the chunk size to allow larger KV caches
        else:
            max_q_chunk_size_effective = 400 if self.mode not in [0,1] else 200 # radically reduce the chunk size to allow larger KV caches
        for start_idx in range(0, q_len, max_q_chunk_size_effective):
            end_idx = min(start_idx + max_q_chunk_size_effective, q_len)

            # Slice the query states for the current chunk
            query_states_chunk = query_states[:, :, start_idx:end_idx, :]

            # Compute attention weights for the chunk
            attn_weights_chunk = torch.matmul(query_states_chunk, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, start_idx:end_idx, :]
                attn_weights_chunk = attn_weights_chunk + causal_mask

            #-----------------------------Top-K * TH Implementation ----------------------------------------
            if self.placement == 'pre-softmax':
                max_attention_scores_chunk = attn_weights_chunk.max(dim=-1)[0].to(dtype=torch.float32) if self.sdc != 'none' and self.sdc_scale > 0.0 else None  # DO THIS BEFORE SOFTMAX and only if SDC compensation is necessary, otherwise skip this computation
                attn_weights_chunk, attn_scores_unselected_chunk, attn_top_mask = self.topk_or_threshold(attn_weights_chunk, start_idx, end_idx)
                existing_denoms_chunk = memory_efficient_sum_exp_attn_minus_max(attn_weights_chunk, max_attention_scores_chunk, dtype=torch.float32)  #if max_attention_scores_chunk is not None else None  # DO THIS BEFORE SOFTMAX compute the per-row denominators of the softmax, considering only the top (selected) elements. Do this step only if SDC compensation is necessary, otherwise skip this computation
                attn_weights_chunk = nn.functional.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32)
                attn_weights_chunk = self.softmax_denominator_compensation(attn_weights_chunk, attn_scores_unselected_chunk, attn_top_mask, existing_denoms_chunk, max_attention_scores_chunk, start_idx, end_idx, inference_phase, dtype=torch.float32).to(query_states.dtype)
            elif self.placement == 'post-softmax':
                attn_weights_chunk = nn.functional.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_chunk, _, _ = self.topk_or_threshold(attn_weights_chunk, start_idx, end_idx)
            elif self.placement == 'none': 
                attn_weights_chunk = nn.functional.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(query_states.dtype)     
                self.dump_stats_attn_elem_and_v_row_full(attn_weights_chunk.size(), inference_phase, start_idx, end_idx)
            else: 
                raise ValueError(f'Illegal topk placement encountered: "{self.placement}"')
            #-----------------------------------------------------------------------------------------

            # Accumulate attention weights if required
            if output_attentions:
                attn_weights.append(attn_weights_chunk)  # TODO change this list append to appending across the q_len dimension

            # Compute attention output for the chunk
            attn_output_chunk = torch.matmul(attn_weights_chunk, value_states)

            if self.vmc:
                attn_output_chunk = self.v_mean_compensation(attn_output_chunk, attn_weights_chunk, value_states, start_idx, end_idx)

            # Store the output for the current chunk
            attn_output[:, :, start_idx:end_idx, :] = attn_output_chunk
        
        if self.dump_qkv:
            self.dump_qkv_to_file(query_states, key_states, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def dump_qkv_to_file(self, query_states, key_states, value_states):
        kv_seq_len = key_states[0].shape[-2]
        q_len = query_states.shape[-2]
        inference_phase = "prefill" if q_len == kv_seq_len else "generative_decoding"

        tensor_list = [query_states, key_states, value_states]
        torch.save(tensor_list, f"{self.products_dir_path}/QKV_{inference_phase}_L{self.id}_seqlen{kv_seq_len}.pt") 

    def get_threshold(self, head_id, seq_len: int, normalize=False) -> float:
        """
        look-up the threshold of the attention head number <head_id> associated
        with the closest sequence length from calibrated set. 

        normalize=True normalizes the threshold by the number of samples (use it 
        when thresholds are needed during the calibration and therefore are still 
        not normalized)
        """
        threshold = self.th_list[head_id].get(seq_len, None)

        if threshold is None:
            closest_seq_len = min(self.th_list[head_id].keys(), key = lambda key: abs(key-seq_len))
            threshold = self.th_list[head_id][closest_seq_len]

        if normalize:
            threshold = aggregate_threshold_list(threshold, self.calib_add_sigma)

        return threshold

    def get_sdc_value(self, head_id, seq_len: int) -> float:
        """
        look-up the sdc (softmax denominator compensation) term of the attention
        head number <head_id> associated with the closest sequence length from 
        calibrated set. 

        normalize=True normalizes the sdc terms by the number of samples (use it 
        when thresholds are needed during the calibration and therefore are still 
        not normalized)
        """
        sdc_term = self.sdc_list[head_id].get(seq_len, None)

        if sdc_term is None:
            closest_seq_len = min(self.sdc_list[head_id].keys(), key = lambda key: abs(key-seq_len))
            sdc_term = self.sdc_list[head_id][closest_seq_len]

        return sdc_term

    def sample_rowids(self, inference_phase, row_sample_fraction:float, seq_len, k) -> List[float]:
        """
        Returns chosen at random list of row indices in [0,seq_len - k) 
        The number of indices is equal to row_sample_fraction * seq_len
        """

        if inference_phase == "generative_decoding":
            sampled_row_th_rowids = [0] if self.rng.random() < row_sample_fraction else []
        elif inference_phase == "prefill":        
            sample_population = list(range(0, seq_len - k))
            if 0 < row_sample_fraction < 1.0:
                row_sample_size = math.ceil(len(sample_population) * 0.5)
                row_obtained_counts = [self.th_num_samples[0].get(l + k + 1, 0) for l in sample_population]
                row_desired_counts = max(row_obtained_counts) + 1 - np.array(row_obtained_counts)
                row_sample_prob = row_desired_counts / row_desired_counts.sum() if row_desired_counts.sum() > 0 else np.ones_like(row_desired_counts)/len(row_desired_counts)
                sampled_row_th_rowids = self.rng.choice(sample_population, size=row_sample_size, replace=False, p=row_sample_prob)  # in [0,seq_len-K)
            elif row_sample_fraction == 1.0:
                sampled_row_th_rowids = sample_population
            else: 
                assert False, "Bad fraction. Must be in (0,1.0]"
        else:
            assert False, "Bad inference_phase. Must be in {prefill,generative_decoding}"
        return sampled_row_th_rowids

    def get_threshold_tensor(self, head_start, head_end, seq_len_start, seq_len_end, normalize=False) -> torch.Tensor:
        """
        returns a 2D tensor th of the shape:
        [head_end-head_start+1, row_end-row_start+1] 
        filled with thresholds, where th[head,seq_len] will contain the floating
        point threshold corresponding to <head> and attention row <seq_len-1>
        """
        return torch.Tensor([[self.get_threshold(head_id, seq_len, normalize) for seq_len in range(seq_len_start,seq_len_end+1)] for head_id in range(head_start, head_end+1)])

    def get_sdc_tensor(self, head_start, head_end, seq_len_start, seq_len_end) -> torch.Tensor:
        """
        returns a 2D tensor sdc_tensor of the shape:
        [head_end-head_start+1, row_end-row_start+1] 
        filled with sdc terms, where sdc_tensor[head,seq_len] will contain the floating
        point value corresponding to <head> and attention row <seq_len-1>
        """
        return torch.Tensor([[self.get_sdc_value(head_id, seq_len) for seq_len in range(seq_len_start,seq_len_end+1)] for head_id in range(head_start, head_end+1)])

    def v_mean_compensation(self, 
                            attn_output_chunk: torch.Tensor, 
                            attn_weights_chunk: torch.Tensor, 
                            value_states: torch.Tensor,
                            start_idx: int, 
                            end_idx: int) -> torch.Tensor:
        """
        Apply V-mean compensation on the attn_output matrix. This compensation
        affects only when the attn_weights has rows (3rd dimension) that sum up
        to < 1.0 and therefore the attn_weights*V product misses some of the V-rows.
        This compensation is aimed to approximately add these missing V-rows back.

        Args:
            attn_output_chunk:  product of the softmax output (attn_weights_chunk)
                          multiplied by the V matrix. 
                          shape: (BSZ, NHEADS, q_chunk_len, HEAD_DIM)
            attn_weights_chunk: tensor containing the softmax output, aka attention
                          scores, aka attention probabilities.
                          shape: (BSZ, NHEADS, q_chunk_len, KV_SEQ_LEN)
            value_states: the value matrix
                          shape: (BSZ, NHEADS, KV_SEQ_LEN, HEAD_DIM)
            start_idx:    first row index of the current attention chunk (inclusive)
            end_idx:      last index of the current attention chunk (exclusive)
        Returns:
          attention output tensor of the same shape as before, but with every row 
          added a special HEAD_DIM-long compensation vector
        """
        BSZ, NHEADS, q_chunk_len, HEAD_DIM = attn_output_chunk.shape
        BSZ, NHEADS, kv_seq_len, HEAD_DIM = value_states.shape
        
        preserved_probability_mass = memory_efficient_sum_reduction(attn_weights_chunk, dim=-1)  # sum up each row -> [BSZ, NHEADS, q_chunk_len]
        lost_probability_mass = 1 - preserved_probability_mass  # take the complementary to represent the probability mass that was removed by the sparsification (topk/th) -> [BSZ, NHEADS, kv_seq_len]
        v_mean_rows = value_states.cumsum(dim=2) / torch.arange(start=1, end=kv_seq_len + 1, step=1, device=value_states.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # for every row r in [0,kv_seq_len-1] compute its causal running v_mean[r,:] vector as an average across V[0:r,:].mean(dim=0) -- > [BSZ, NHEADS, kv_seq_len, HEAD_DIM]
        
        if q_chunk_len == 1:  # generative_decoding - use last mean_v row
            v_mean_rows_chunk = v_mean_rows[:,:,-1:,:]
        else:  # chunked prefill - need to carefully pick the correct mean v_rows for compensation
            assert(0 <= start_idx < end_idx <= kv_seq_len)
            v_mean_rows_chunk = v_mean_rows[:,:,start_idx:end_idx,:]

        attn_output_chunk = attn_output_chunk + lost_probability_mass.unsqueeze(-1) * v_mean_rows_chunk  # [BSZ, NHEADS, q_chunk_len, HEAD_DIM] + [BSZ, NHEADS, q_chunk_len, unsqueezed] * [BSZ, NHEADS, q_chunk_len, HEAD_DIM]
        return attn_output_chunk

    def softmax_denominator_compensation(self, 
                                         attn_weights_chunk: torch.Tensor, 
                                         attn_scores_unselected_chunk: torch.Tensor, 
                                         attn_top_mask: torch.Tensor,
                                         existing_denoms_chunk: torch.Tensor,
                                         max_attention_scores_chunk: torch.Tensor,
                                         start_idx:int,
                                         end_idx:int,
                                         inference_phase:str,
                                         dtype=torch.float32,
                                         ) -> torch.Tensor:
        """
        Apply softmax denominator compensaton on the attention weights tensor,
        which is assumed to have undergone the softmax already. The idea is to
        renormalize each of the attn_weights (post-softmax probabilities) by a
        larger denominator that includes the sum of exponents of elements that 
        were filtered out before the softmax took place).

        The precise method of compensation must be set by self.sdc.

        Anyway, only attn_weights's rows > K are compensated!

        Args:
          attn_weights_chunk - attention tensor (after the softmax was applied)
                         Shape: (BATCH_SIZE, NHEADS, q_chunk_len, kv_seq_len)
          attn_scores_unselected_chunk - attention tensor (before the softmax was applied)
                         with -inf in all the places which 
                         were chosen to be kept by topk/thinfinity
                         Shape: (BATCH_SIZE, NHEADS, q_chunk_len, kv_seq_len)
          attn_top_mask - boolean mask with True in places that were selected 
                         to keep. (BATCH_SIZE, NHEADS, q_chunk_len, kv_seq_len)
                         or topk. Shape: (BATCH_SIZE, NHEADS, q_chunk_len, kv_seq_len)
          existing_denoms_chunk - sum of e^{xi-max(x)} (before softmax) per every row
                         of the attetnion matrix *after-top-k/th* but *before-softmax*
                         Shape: (BATCH_SIZE, NHEADS, q_chunk_len)
          max_attention_scores_chunk -  per attention row (per head per batch dim)
                         maximum of the elements (must be computed before the 
                         softmax. i.e. on the attention scores, aka logits) 
                         Shape: (BATCH_SIZE, NHEADS, q_chunk_len)
          start_idx     - index of the first attention row being processed (inclusive)
          end_idx       - index of the last attention row being processed (exclusive)
          inference_phase - str, either "generative_decoding" or "prefill"
          dtype         - data type, in which to process the compensation
        Returns:
          attention matrix of the same shape as before, but with the attention 
          scores renormalized by a larger denominator 
        """
        BATCH_SIZE, NHEADS, q_chunk_len, kv_seq_len = attn_weights_chunk.size()
        K = self.K[self.id] 

        # no compensation required
        if self.sdc == 'none' or self.sdc_scale == 0.0 or kv_seq_len <= K or attn_top_mask is None or (inference_phase == "prefill" and end_idx < K):
            return attn_weights_chunk

        # determine r_chunk - the first attn row (within the attention chunk) tthat the top-k or top-theta were applied from, hence an SDC is required there
        if inference_phase=="generative_decoding":
            r_chunk = 0 # there is only 1 row duing decoding, regardless of chunk size
        else:
            if K < start_idx:  # end_idx <= K: apply top-theta to the entire chunk
                r_chunk = 0
            elif start_idx <= K < end_idx:  # top-theta thresholding needs to be carefully applied from a very specific row in the attention matrix chunk
                r_chunk = K - start_idx
            else: 
                assert(False and "Shouldn't happen due to the above condition of K <= end_idx. The chunk's entire range of rows [start_idx, end_id) is below K, hence no Top-k/theta should take place")
            
        if self.sdc == 'exact':
            assert ((BATCH_SIZE, NHEADS, q_chunk_len, kv_seq_len) == attn_scores_unselected_chunk.size())
            assert ((BATCH_SIZE, NHEADS, q_chunk_len) == existing_denoms_chunk.size())
            assert ((BATCH_SIZE, NHEADS, q_chunk_len) == max_attention_scores_chunk.size())
            missing_denominator_term = memory_efficient_sum_exp_attn_minus_max(attn_scores_unselected_chunk[:,:,r_chunk:,:],
                                                                               max_attention_scores_chunk[:,:,r_chunk:],
                                                                               dtype)
            new_denom = existing_denoms_chunk[:,:,r_chunk:] + self.sdc_scale * missing_denominator_term
            attn_weights_chunk[:,:,r_chunk:,:].mul_(existing_denoms_chunk[:,:,r_chunk:].unsqueeze(-1))
            attn_weights_chunk[:,:,r_chunk:,:].div_(new_denom.unsqueeze(-1))
        elif self.sdc == 'exp-threshold':
            assert self.mode == 0, "exp-threshold compensation is only applicable in mode=0 (thresholding)"
            if not self.calibrate:
                assert ((BATCH_SIZE, NHEADS, q_chunk_len, kv_seq_len) == attn_top_mask.size())
                assert ((BATCH_SIZE, NHEADS, q_chunk_len) == max_attention_scores_chunk.size())

                num_selected_elements = memory_efficient_sum_reduction(attn_top_mask, dim=-1)
                num_unselected_elements = (kv_seq_len - num_selected_elements)
                
                # look-up the SEQ_LEN-K thresholds from the closest sequence length from calibrated set. 
                per_head_row_thresholds = self.get_threshold_tensor(0, NHEADS - 1, max(start_idx, K), end_idx-1)  # [NHEADS, SEQ_LEN-K] - a 2D matrix of per-head-per-row thresholds 
                per_head_row_thresholds = per_head_row_thresholds.to(device=attn_weights_chunk.device, dtype=attn_weights_chunk.dtype)

                # Renormalize softmax score (e^a_i/existing_denoms[row]) by multiplying
                # it by "existing_denoms[row]/(existing_denoms[row] + sdc_scale * num_unselected[row] * e^(th - max(a))"
                missing_denominator_term = num_unselected_elements[:,:,r_chunk:] * torch.exp(per_head_row_thresholds.unsqueeze(0) - max_attention_scores_chunk[:,:,r_chunk:].to(dtype=existing_denoms_chunk.dtype))
                new_denom = existing_denoms_chunk[:,:,r_chunk:] + self.sdc_scale * missing_denominator_term
                attn_weights_chunk[:,:,r_chunk:,:] = attn_weights_chunk[:,:,r_chunk:,:].mul(existing_denoms_chunk[:,:,r_chunk:].unsqueeze(-1)).div(new_denom.unsqueeze(-1))

        elif self.sdc == 'offline-calibrated':
            assert self.mode in {0,1}, "offline-calibrated compensation is available in mode 0 (thresholding) and 1 (topk)"
            if not self.calibrate:

                # look-up per-row missing denominator terms from the calibrated dictionary
                missing_denominator_terms = self.get_sdc_tensor(0, NHEADS - 1, max(start_idx, K), end_idx-1)  # [NHEADS, SEQ_LEN-K] - a 2D matrix of per-head-per-row calibrated sdc terms 
                missing_denominator_terms = missing_denominator_terms.to(dtype=attn_weights_chunk.dtype, device=attn_weights_chunk.device)

                # Renormalize softmax score of row vector a "e^(a_i-max(a))/existing_denoms[row]" by multiplying
                # it by "existing_denoms[row]/(existing_denoms[row] + sdc_scale * missing_denominator_term_from_calibration[row])"
                new_denom = existing_denoms_chunk[:,:,r_chunk:] + self.sdc_scale * missing_denominator_terms.unsqueeze(0).unsqueeze(0)
                attn_weights_chunk[:,:,r_chunk:,:]  = attn_weights_chunk[:,:,r_chunk:,:] .mul(existing_denoms_chunk[:,:,r_chunk:].unsqueeze(-1)).div(new_denom.unsqueeze(-1))

        else:
            assert NotImplementedError(f"self.sdc compensation is not supported")

        return attn_weights_chunk

    def topk_or_threshold(
            self,
            attn_weights: torch.Tensor,
            start_idx: int,
            end_idx: int            
        ) -> torch.Tensor:
        """
        apply the top-k or the thresholding (according to the self.mode) on 
        the attn_weights tensor across its last dimension. This results in 
        keeping only a limited number of elements within each row of each head's
        attention weight matrix)

        Arguments
            attn_weights - "chunk of attention weights" 
                           Tensor of shape (bsz, self.num_heads, q_chunk_len, kv_seq_len).
                           This is the primary input tensor to be processed
            start_idx    - Integer specifyng the first attention row id (inclusive) 
                           that should be computed used for chunked prefill
            end_idx      - Integer specifyng the last attention row id (excluisve) 
                           that should be computed used for chunked prefill

        Returns: 3 tensors:
                 1) the attn_weights tensor (bsz, self.num_heads, q_chunk_len, kv_seq_len) 
                 after the application of Top-k / thresholding.
                 2) attn_weights_unselected - the complementary tensor to attn_weights,
                    where the not selected weights are equal to original weight, whereas the
                    selected weights are replaced by 0 or -inf. Will be None if self.sdc is 'none'
                 3) attn_top_mask - boolean mask signifying the elements that were selected
                 to be kept by topk/th. Will be None if self.sdc is 'none'
        """
        BATCH_SIZE, NHEADS, q_chunk_len, kv_seq_len = attn_weights.size()
        assert (q_chunk_len == end_idx - start_idx)
        K = self.K[self.id] if isinstance(self.K, list) else self.K
        inference_phase = "generative_decoding" if q_chunk_len == 1 else "prefill"
        attn_scores_unselected = None
        attn_top_mask = None
    
        # Value to replace the filtered-out (non-topk, <= threshold) elements with
        if self.mode in {0,1}:
            if self.placement == 'pre-softmax':
                # Before softmax - replacement value should be as low as possible: negative infinity
                replacement_value = torch.finfo(attn_weights.dtype).min
            elif self.placement == 'post-softmax':
                # After softmax - the replacement value is the minimum of the softmax range: zero
                replacement_value = 0.0
            else:
                raise ValueError("Invalid placement for a topk/threshold.")

        if not self.calibrate:
            if self.id==0:
                if inference_phase == "prefill":
                    with open(f"{self.products_dir_path}/sequence_lengths_per_example.csv",'a') as f:
                        f.write(f'{kv_seq_len}\n')                    
            
            # ----- mode-0 Thresholding -----
            if self.mode == 0:

                # Initialize selection mask and unselected scores
                if inference_phase=="prefill":
                    attn_top_mask = lower_triangular_slice(start_idx, end_idx, kv_seq_len, device=attn_weights.device) 
                    attn_top_mask = attn_top_mask.unsqueeze(0).repeat(BATCH_SIZE, 1, 1).unsqueeze(1).repeat(1, NHEADS, 1, 1)
                else:
                    attn_top_mask = torch.full(attn_weights.size(), True, dtype=torch.bool, device=attn_weights.device)
                attn_scores_unselected = torch.full_like(attn_weights, replacement_value) if self.sdc != 'none' and self.sdc_scale > 0.0 else None  # cancel allocation when no further use of this tensor will be made 
                
                if 0 < K < kv_seq_len and (inference_phase != "prefill" or K <= end_idx):
                    # look up for the claibrated per-attn-row (per-sequence-length) thresholds
                    if inference_phase=="generative_decoding":
                        per_head_row_thresholds = self.get_threshold_tensor(0, NHEADS - 1, kv_seq_len, kv_seq_len)  # [NHEADS,1] at generative decoding - a 2D matrix of per-head-per-row thresholds 
                    else:
                        per_head_row_thresholds = self.get_threshold_tensor(0, NHEADS - 1, max(start_idx, K), end_idx - 1)  # [NHEADS, chunk_size-K or chunk_size] at prefill- a 2D matrix of per-head-per-row thresholds 
                    per_head_row_thresholds = per_head_row_thresholds.to(device=attn_weights.device, dtype=attn_weights.dtype)

                    # determine r_chunk - the first attn row (within the attention chunk) to threshold from it until the last rows (dim=2) of attn_weights
                    if inference_phase=="generative_decoding":
                        r_chunk = 0 # there is only 1 row duing decoding, regardless of chunk size
                    else:
                        if K < start_idx:  # end_idx <= K: apply top-theta to the entire chunk
                            r_chunk = 0
                        elif start_idx <= K < end_idx:  # top-theta thresholding needs to be carefully applied from a very specific row in the attention matrix chunk
                            r_chunk = K - start_idx
                        else: 
                            assert(False and "Shouldn't happen due to the above condition of K <= end_idx. The chunk's entire range of rows [start_idx, end_id) is below K, hence no Top-k/theta should take place")
                        assert(q_chunk_len - r_chunk == per_head_row_thresholds.shape[-1])  # verify that enough per-seq-len thresholds were preapred for the relevant attention rows

                    # Apply threholding - set a bitmask of items to be kept (>th)
                    attn_top_mask[:,:,r_chunk:,:] = torch.gt(attn_weights[:,:,r_chunk:,:], per_head_row_thresholds.unsqueeze(0).unsqueeze(-1))

                    # capk - keep at most K last elements in every row
                    if self.capk:
                        cumsum = attn_top_mask[:,:,r_chunk:,:].cumsum(dim=-1)
                        cumsum_rev = cumsum.max(dim=-1, keepdim=True)[0] - cumsum
                        attn_top_mask[:,:,r_chunk:,:] = attn_top_mask[:,:,r_chunk:,:] & (cumsum_rev < K)
                    
                    if attn_scores_unselected is not None:
                        attn_scores_unselected[:,:,r_chunk:,:] = torch.where(attn_top_mask[:,:,r_chunk:,:], replacement_value, attn_weights[:,:,r_chunk:,:])
                    attn_weights[:,:,r_chunk:,:] = torch.where(attn_top_mask[:,:,r_chunk:,:], attn_weights[:,:,r_chunk:,:], replacement_value)
               
                
                # Collect relative (to topk method) number of attention 
                # elements that survived the thresholding.
                # buff_occupancy_relative should be ~ 1 for good thresholding.
                buff_occupancy_topk_per_head = BATCH_SIZE * self.compute_buff_occupancy_topk_per_head(inference_phase, K, kv_seq_len, start_idx, end_idx)
                buff_occupancy_total_per_head = memory_efficient_sum_reduction(attn_top_mask, dim=(2,3))
                buff_occupancy_relative_per_head = buff_occupancy_total_per_head / buff_occupancy_topk_per_head
                with open(f"{self.products_dir_path}/layer{self.id}.txt",'a') as f:
                    for b_, h_ in itertools.product(range(BATCH_SIZE), range(NHEADS)):
                        f.write(f'L{self.id}_H{h_}:{kv_seq_len} {K} {inference_phase} {buff_occupancy_relative_per_head[b_, h_]}\n')
            
            # ----- mode-1 TopK -----               
            if self.mode == 1:
                # Initialize selection mask and unselected scores
                if inference_phase=="prefill":
                    attn_top_mask = lower_triangular_slice(start_idx, end_idx, kv_seq_len, device=attn_weights.device) 
                    attn_top_mask = attn_top_mask.unsqueeze(0).repeat(BATCH_SIZE, 1, 1).unsqueeze(1).repeat(1, NHEADS, 1, 1)
                else:
                    attn_top_mask = torch.full(attn_weights.size(), True, dtype=torch.bool, device=attn_weights.device)
                attn_scores_unselected = torch.full_like(attn_weights, replacement_value) if self.sdc != 'none' and self.sdc_scale > 0.0 else None  # cancel allocation when no further use of this tensor will be made 
                if 0 < K < kv_seq_len and (inference_phase != "prefill" or K <= end_idx):
                    # Find Top-k elements per row in attention rows [K, K+1,...]:
                    # determine the first attn row (within the attention chunk) to apply top-k from it until the last rows (dim=2) of attn_weights
                    if inference_phase=="generative_decoding":
                        r_chunk = 0 # there is only 1 row during decoding, regardless of chunk size
                    else:
                        if K < start_idx:  # end_idx <= K: apply top-theta to the entire chunk
                            r_chunk = 0
                        elif start_idx <= K < end_idx:  # top-theta thresholding needs to be carefully applied from a very specific row in the attention matrix chunk
                            r_chunk = K - start_idx
                        else: 
                            assert(False and "Shouldn't happen due to the above condition of K <= end_idx. The chunk's entire range of rows [start_idx, end_id) is below K, hence no Top-k/theta should take place")
                        # r_chunk = 0 if start_idx > K else K 

                    vals, idxs = attn_weights[:,:,r_chunk:,:].topk(K, dim=-1)
                    attn_top_mask[:,:,r_chunk:,:].fill_(False).scatter_(-1, idxs, True)   
                    if attn_scores_unselected is not None:
                        attn_scores_unselected[:,:,r_chunk:,:] = torch.where(attn_top_mask[:,:,r_chunk:,:], replacement_value, attn_weights[:,:,r_chunk:,:])
                    attn_weights[:,:,r_chunk:,:] = torch.where(attn_top_mask[:,:,r_chunk:,:], attn_weights[:,:,r_chunk:,:], replacement_value)

            # Write statistics - number of kept attention elements and number of reuqired V rows
            if not self.calibration_phase: # double check that we are not in calibration sample (can happen after the prefill phase of the last calibration token is done)
                if attn_top_mask is None:
                    self.dump_stats_attn_elem_and_v_row_full([BATCH_SIZE, NHEADS, q_chunk_len, kv_seq_len], inference_phase, start_idx, end_idx)
                else:
                    self.dump_stats_attn_elem_and_v_row_from_mask(attn_top_mask, inference_phase, start_idx, end_idx)
                    
            if attn_scores_unselected is None:
                attn_top_mask = None

        else:
            # ----- Performing Calibration -----
            assert(False and "TODO: verify the entire calibration after we introduced chunked prefill")
            sampled_row_th_rowids = []
            assert(self.num_calib_requests != 0), f"number of calibration requests was not set"
            r_chunk = 0 if inference_phase=="generative_decoding" else K # first attn row to apply thresholding/topk from it until the last rows (dim=2) of attn_weights
            
            if self.mode == 0 and 0 < K < kv_seq_len:
                # -- calibration of thresholds for top-th --
                # Thresholding based on Top-K:

                # Step 1 - find multiple thresholds: for a list of ks
                if self.reduce_gpu_mem:
                    # chunked (several heads per chunk) quantile computation on gpu to keep the memory requirements low
                    quant_chunks = []
                    for attn_heads_chunk in torch.tensor_split(attn_weights[:,:,r_chunk:,:], 4, dim=1):
                        quant_chunk = torch.quantile(attn_heads_chunk.float(), 1 - K / kv_seq_len , dim=3, interpolation='lower')  # quantile() requires the input tensor dtype to be either float or double
                        quant_chunks.append(quant_chunk)
                    quant = torch.cat(quant_chunks, dim=1) 
                else:
                    quant = torch.quantile(attn_weights[:,:,r_chunk:,:].float(), 1 - K / kv_seq_len , dim=3, interpolation='lower')  
                # quant tensor (NUM_BATCH_SIZE, NHEADS, kv_seq_len)
                # contains a threshold per row of attention matrix (per head per batch example) 

                # Step 2 - record the thresholds
                # sample <calib_sample_frac> of rows to actually calibrate on (bias towards the less sampled ones so far)
                sampled_row_th_rowids = self.sample_rowids(inference_phase, self.calib_sample_frac, kv_seq_len, K)  # row indices in ~ U[0, seq_len - k) for prefill, and either [0] or [] for generative_decoding
                for sample_id, head_id, quant_row_id in itertools.product(range(BATCH_SIZE), range(NHEADS), sampled_row_th_rowids):
                    rowid_per_row_exp_sums = quant_row_id + r_chunk  # attention row corresponding to this quant row
                    row_seq_len = rowid_per_row_exp_sums + 1 if inference_phase=="prefill" else kv_seq_len  # the sequence length corresponding to this threshold row
                    row_th = quant[sample_id, head_id, quant_row_id].tolist() # list containing row_seqlen-1 thresholds
    
                    # Record the row_th that was determined for this head
                    if row_seq_len not in self.th_list[head_id]:
                        # self.th_list[head_id][row_seq_len] = row_th
                        self.th_list[head_id][row_seq_len] = [row_th]
                        self.th_num_samples[head_id][row_seq_len] = 1
                    else:
                        # self.th_list[head_id][row_seq_len] += row_th
                        self.th_list[head_id][row_seq_len].append(row_th)
                        self.th_num_samples[head_id][row_seq_len] += 1                
            
                # DELETE ME IF TH_LOG.TXT is NOT NEEDED
                # with open(f"{self.products_dir_path}/th_log.txt",'a') as f:
                #     for head_id, quant_row_id in itertools.product(range(NHEADS), sampled_row_th_rowids):
                #         rowid_attn = quant_row_id + K 
                #         row_seq_len = rowid_attn + 1
                #         f.write(f'L{self.id}_H{head_id} {K} {row_seq_len} {self.th_list[head_id][row_seq_len][-1]}\n')

            if self.sdc == 'offline-calibrated' and 0 < K < kv_seq_len:
                # -- calibration for softmax denominator compensation --
                sampled_row_th_rowids = self.sample_rowids(inference_phase, self.calib_sample_frac, kv_seq_len, K) if len(sampled_row_th_rowids) == 0 else sampled_row_th_rowids # row indices in ~ U[0, seq_len - k) - for prefill; either [0] or [] for generative-decoding
                assert replacement_value == torch.finfo(attn_weights.dtype).min  # implicitly validates pre-softmax placement
                if self.mode == 0:
                    #  apply the opposite of the top-th (keep below-or-equal to threshold)- with the so-far calibrated threshold 
                    per_head_row_thresholds = self.get_threshold_tensor(0, NHEADS - 1, max(kv_seq_len-q_chunk_len+1,K+1), kv_seq_len, normalize=True)  # [NHEADS, SEQ_LEN-K] - a 2D matrix of per-head-per-row thresholds. Also normalize! because the thresholds are now only sums (calibration ongoing)
                    per_head_row_thresholds = per_head_row_thresholds.to(device=attn_weights.device, dtype=attn_weights.dtype)
                    unselected_attn_weights = torch.where(attn_weights[:,:,r_chunk:,:] <= per_head_row_thresholds.unsqueeze(0).unsqueeze(-1), 
                                                          attn_weights[:,:,r_chunk:,:], 
                                                          replacement_value).to(torch.float32)

                elif self.mode == 1:
                    #  apply bottom-(N-k), check how many elements left per row                   
                    vals, idxs = attn_weights[:,:,r_chunk:,:].topk(kv_seq_len-K, dim=-1, largest=False, sorted=False) # Non-top-k <==> Bottom-N-K
                    unselected_attn_weights = torch.full(attn_weights[:,:,r_chunk:,:].size(), 
                                                         replacement_value, 
                                                         dtype=attn_weights.dtype,
                                                         device=attn_weights.device).scatter_(-1, idxs, vals).to(torch.float32)
                else:
                    assert False, "--sdc 'offline-calibrated' is only allowed for mode=0 or 1."
                
                # find sum(exp(a_i - max(a))) across each row vector "a", where a_i are non-top-k / below-threshold elements. Note that the max(a) s taken across all the elements (including the kept once)
                per_row_exp_maxes = attn_weights[:,:,r_chunk:,:].max(dim=-1, keepdims=True)[0]
                per_row_exp_sums = memory_efficient_sum_reduction(torch.exp(unselected_attn_weights - per_row_exp_maxes), dim=-1).to(attn_weights.dtype)  # [BATCH_SIZE, NHEADS, SEQ_LEN-K]
                # per_row_avg_exp_sum = per_row_exp_sums.view([BATCH_SIZE * NHEADS, SEQ_LEN - K]).mean(0)  # for every token position (attenton row) find an average
                for sample_id, head_id, sampled_row_id in itertools.product(range(BATCH_SIZE), range(NHEADS), sampled_row_th_rowids):
                    row_seq_len = sampled_row_id + r_chunk + 1 if inference_phase=="prefill" else kv_seq_len
                    exp_sum = per_row_exp_sums[sample_id,head_id,sampled_row_id].item()
                    if row_seq_len not in self.sdc_list[head_id]:
                        self.sdc_list[head_id][row_seq_len] = [exp_sum]
                        self.sdc_num_samples[head_id][row_seq_len] = 1
                    else:
                        self.sdc_list[head_id][row_seq_len].append(exp_sum)
                        self.sdc_num_samples[head_id][row_seq_len] += 1

            if self.calib_tac and self.mode == 0 and 0 < K < kv_seq_len:
                # process the attn_weights as if top-k was performed. This 
                # should help the subsequence layers of the model to calibrate 
                # on a more accurately represented (sparsified) activations
                vals, idxs = attn_weights[:,:,r_chunk:,:].topk(K, dim=-1)
                attn_weights[:,:,r_chunk:,:] = torch.full(attn_weights[:,:,r_chunk:,:].size(), 
                                                    replacement_value, 
                                                    dtype=attn_weights.dtype, 
                                                    device=attn_weights.device).scatter_(-1, idxs, vals)

            if inference_phase ==  "prefill":
                self.obt_calib_requests += 1

            ################################################################################################
            ####### TODO, replace the following condition with external call for dumping th.txt         ####
            #######       this will avoid the need to reach exactly the anticipated num_calib_requests  ####
            #######       which is extremely useful since lm_eval v0.4.8, the number of requests that   ####
            #######       invoke model.forward() through lm_eval.evaluate() - is npot guaranteed due    ####
            #######       to some internal caching, which skips forwad() from time to time              ####
            ################################################################################################
            if self.obt_calib_requests == self.num_calib_requests:
                if self.mode == 0:
                    # Finalize the calibrated threshold by aggregating its calibration samples using aggregate_threshold_list(samples)
                    for head_id in range(NHEADS):
                        if len(self.th_list[head_id]) == 0:
                            raise ValueError(f"{type(self).__name__} after the calibration no thresholds were recorded "
                                             f"in layer {self.id} (k={K}, num_calib_requests={self.num_calib_requests}). "
                                             "It is possible that all the observed calibration samples had sequence "
                                             f"length below k={K}. Suggestion: reduce k or increase num_calib_requests)")
                        self.th_list[head_id] = {seqlen: aggregate_threshold_list(row_th_lst, self.calib_add_sigma) for seqlen, row_th_lst in sorted(self.th_list[head_id].items())}

                    # Dump thresholds to a file
                    with open(f"{self.products_dir_path}/th.txt",'a') as f:
                        for head_id in range(self.num_heads):
                            for seqlen, th in self.th_list[head_id].items():
                                f.write(f'L{self.id}_H{head_id}:{seqlen} {th} {self.th_num_samples[head_id][seqlen]} {K}\n')
                    print(f"--Calibration (threshold) done for layer-{self.id}")

                if self.sdc == 'offline-calibrated':
                    # for every observed sequence length - the compensation term
                    # to keep is the average across calibration samples
                    for head_id in range(NHEADS):
                        self.sdc_list[head_id] = {seqlen: aggregate_threshold_list(row_sdc_lst) for seqlen, row_sdc_lst in sorted(self.sdc_list[head_id].items())}

                    # Dump compensation terms to a file
                    with open(f"{self.products_dir_path}/sdc.txt",'a') as f:
                        for head_id in range(self.num_heads):
                            for seqlen, sdc in self.sdc_list[head_id].items():
                                f.write(f'L{self.id}_H{head_id}:{seqlen} {sdc} {self.sdc_num_samples[head_id][seqlen]}\n')
                    print(f"--Calibration (sdc) done for layer-{self.id}")
                
                self.calibrate = False # prevent further calibration (important when there are some generataive decoding passes that will still be invoked)

        return attn_weights, attn_scores_unselected, attn_top_mask

    def triangle_area_incl_diagonal(self, side_length:int) -> int:
            """
            compute an area of a lower trinagle matrix including the main diagonal
            """
            return side_length * (side_length + 1) / 2
    
    def compute_buff_occupancy_topk_per_head(self, inference_phase:str, k:int, kv_seq_len:int, start_idx:int, end_idx:int) -> int:
        """
        return the number number of attention elements that the Top-k-Attention method would keep per attention head
        considering that the attention is **chunked during the prefill** i.e. taking only the rows [start_idx, end_idx)
        """

        # the row idx k-1 is an important row id, because it corresponds to 
        # kv_seq_len = k, hence it's the last row where top-k attnetion will 
        # keep all elements
        k_idx = k - 1 

        if 0 < k <= kv_seq_len:
            # topk-k will take all elements only until row k_idx (inclusive), after which it will take only k
            if inference_phase=="generative_decoding":
                buff_occupancy_topk_per_head = k
            else:
                assert (start_idx < end_idx)
                if end_idx <= k_idx:
                    # take the entire lower triangle (triangle until row "end_idx" minus triangle of previous chunks)
                    triangle_until_end_idx = self.triangle_area_incl_diagonal(end_idx)
                    trinangle_of_previous_chunks = self.triangle_area_incl_diagonal(start_idx) # note: row start_idx is excluded from this area
                    buff_occupancy_topk_per_head = triangle_until_end_idx - trinangle_of_previous_chunks
                if start_idx < k_idx < end_idx:
                    above_k = self.triangle_area_incl_diagonal(k_idx) - self.triangle_area_incl_diagonal(start_idx) # difference between a triangle that ends at row k-1 (exclusive) and the triangle area of previous chunks (before start_idx - exclusive)
                    below_k = (end_idx - k_idx) * k  # rectangle that includes tow k_idx, ..., end_idx-1
                    buff_occupancy_topk_per_head = above_k + below_k
                else: # k_idx <= start_idx: take a rectangle
                    buff_occupancy_topk_per_head = (end_idx - start_idx) * k
        else:
            # easy case when no topk can even be applied (sequence length shorter than k)
            if inference_phase=="generative_decoding":
                buff_occupancy_topk_per_head = kv_seq_len 
            else:
                # take the entire lower triangle and subtract the previous chunks' triangle
                triangle_until_end_idx = self.triangle_area_incl_diagonal(end_idx)
                trinangle_of_previous_chunks = self.triangle_area_incl_diagonal(start_idx)
                buff_occupancy_topk_per_head = triangle_until_end_idx - trinangle_of_previous_chunks
        
        return buff_occupancy_topk_per_head  


    def dump_stats_attn_elem_and_v_row_full(self, attn_top_mask_shape: Tuple[int,int,int,int], inference_phase:str, start_idx:int, end_idx:int):
        """
        Assuming that the entire causal matrix has been processed,
        write 2 statistics files per-layer
            <products_dir_path>/layer<id>_kept_attn_<inference_phase>.csv (per-head statistics)
            <products_dir_path>/layer<id>_kept_vrow_<inference_phase>.csv (per group statistics)
        """       
        batch_size, num_heads, q_chunk_len, kv_seq_len = attn_top_mask_shape
        assert(inference_phase!="prefill" or 1 < q_chunk_len <= self.max_q_chunk_size)
        assert(inference_phase!="generative_decoding" or q_chunk_len == 1)

        # per-head attention elements count
        if f'kept_attn_{inference_phase}' in self.dump_stats_set:
            full_attn_numel_one_head = batch_size * kv_seq_len if inference_phase=="generative_decoding" else batch_size * (self.triangle_area_incl_diagonal(end_idx) - self.triangle_area_incl_diagonal(start_idx))  # causal full attn matrix within the chunk rows [start_idx,end_idx)
            with open(f"{self.products_dir_path}/layer{self.id}_kept_attn_{inference_phase}.csv",'a') as f:
                for b_, h_ in itertools.product(range(batch_size), range(num_heads)):
                    # layer head kv-seq-len kept_attn_numel_per_head full_attn_numel_one_head
                    f.write(f'{self.id},{h_},{kv_seq_len},{full_attn_numel_one_head},{full_attn_numel_one_head}\n')          

        # per-group V-row read count     
        if f'kept_vrow_{inference_phase}' in self.dump_stats_set:
            full_vrow_num_per_group = kv_seq_len if inference_phase=="generative_decoding" else end_idx
            with open(f"{self.products_dir_path}/layer{self.id}_kept_vrow_{inference_phase}.csv",'a') as f:
                for b_, g_ in itertools.product(range(batch_size), range(self.num_key_value_heads)):  #num_key_value_heads is actually key-value groups of query heads (each group containns num_key_value_groups query heads associated to 1 kv_head)
                    # layer group kv-seq-len kept_vrow_num_per_group full_vrow_num_per_group
                    f.write(f'{self.id},{g_},{kv_seq_len},{full_vrow_num_per_group},{full_vrow_num_per_group}\n') 


    def dump_stats_attn_elem_and_v_row_from_mask(self, attn_top_mask: torch.Tensor, inference_phase:str, start_idx:int, end_idx:int):
        """
        Assuming that only the selected elements of the attention matrix have been processed,
        write 3 statistics files per-layer 
            <products_dir_path>/layer<id>_kept_attn_<inference_phase>.csv (per-head statistics)
            <products_dir_path>/layer<id>_kept_vrow_<inference_phase>.csv (per group statistics)
            <products_dir_path>/layer<id>_kept_vrow_popularities_<inference_phase>.txt" (per group statistics)
        """
        
        batch_size, num_heads, q_chunk_len, kv_seq_len = attn_top_mask.size() 

        assert(inference_phase!="prefill" or 1 < q_chunk_len <= self.max_q_chunk_size)
        assert(inference_phase!="generative_decoding" or q_chunk_len == 1)

        # per-head attention elements count
        if f'kept_attn_{inference_phase}' in self.dump_stats_set:
            kept_attn_numel_per_head = memory_efficient_sum_reduction(attn_top_mask, dim=(2,3))  # returns a tensor of counters, shape: [B,NH]
            full_attn_numel_one_head = batch_size * kv_seq_len if inference_phase=="generative_decoding" else batch_size * (self.triangle_area_incl_diagonal(end_idx) - self.triangle_area_incl_diagonal(start_idx))  # causal full attn matirx within the chunk rows [start_idx,end_idx)
            with open(f"{self.products_dir_path}/layer{self.id}_kept_attn_{inference_phase}.csv",'a') as f:
                for b_, h_ in itertools.product(range(batch_size), range(num_heads)):
                    # layer head kv-seq-len kept_attn_numel_per_head full_attn_numel_one_head
                    f.write(f'{self.id},{h_},{kv_seq_len},{kept_attn_numel_per_head[b_, h_]},{full_attn_numel_one_head}\n')        

        # per-group V-row popularity counters (each line - <kv_seq_len> popularity counters)
        if f'kept_vrow_popularities_{inference_phase}' in self.dump_stats_set or f'kept_vrow_{inference_phase}' in self.dump_stats_set:
            popcount_vrow_per_head = memory_efficient_sum_reduction(attn_top_mask, dim=2)  # [B,NH,kv_seq_len] for every v-row index - count how many attention rows need it
            popcount_vrow_per_head_grouped = popcount_vrow_per_head.reshape(batch_size, self.num_key_value_heads, self.num_key_value_groups, kv_seq_len)  # [B,NHKV,G,kv_seq_len]
            popcount_vrow_per_group = popcount_vrow_per_head_grouped.sum(dim=2)  # [B,NHKV,kv_seq_len]
            if f'kept_vrow_popularities_{inference_phase}' in self.dump_stats_set: 
                with open(f"{self.products_dir_path}/layer{self.id}_kept_vrow_popularities_{inference_phase}.txt",'a') as f:
                    for b_, g_ in itertools.product(range(batch_size), range(self.num_key_value_heads)):  #num_key_value_heads is actually key-value groups of query heads (each group containns num_key_value_groups query heads associated to 1 kv_head)
                        # layer group kv-seq-len comma-separated-per-v-row-id-counts-of-popularities
                        f.write(f'{self.id},{g_},{kv_seq_len},{popcount_vrow_per_group[b_, g_].tolist()}\n')      

            # per-group V-row read count     
            if f'kept_vrow_{inference_phase}' in self.dump_stats_set:
                kept_vrow_num_per_group = popcount_vrow_per_group.count_nonzero(dim=2)  # [B,NHKV]
                full_vrow_num_per_group = kv_seq_len if inference_phase=="generative_decoding" else end_idx        
                with open(f"{self.products_dir_path}/layer{self.id}_kept_vrow_{inference_phase}.csv",'a') as f:
                    for b_, g_ in itertools.product(range(batch_size), range(self.num_key_value_heads)):  #num_key_value_heads is actually key-value groups of query heads (each group containns num_key_value_groups query heads associated to 1 kv_head)
                        # layer group kv-seq-len kept_vrow_num_per_group full_vrow_num_per_group
                        f.write(f'{self.id},{g_},{kv_seq_len},{kept_vrow_num_per_group[b_, g_]},{full_vrow_num_per_group}\n') 


# %% Update the Vanilla model with Top-K layers
count=0
def update_model(model, reduce_gpu_mem, products_dir_path):
    global count
    for child_name, child in model.named_children():
        if isinstance(child, LlamaDecoderLayer):
            attention = child.self_attn
            # import copy
            # attention_copy = copy.deepcopy(attention)
            topk_attention = TopK_LLamaAttention(attention.config, count, reduce_gpu_mem, products_dir_path)
            
            # Copy params and load model to the same device
            device = next(attention.parameters()).device
            topk_attention.load_state_dict(attention.state_dict())
            topk_attention.to(device)
            topk_attention.eval()
            count += 1
            # child.self_attn = attention_copy
            child.self_attn = topk_attention
            # pass
        else:
            update_model(child, reduce_gpu_mem, products_dir_path)
            
def load_thresholds_from_directory(calib_load_path: str, layer:int, num_heads:int, copy_to:str = None, verbose:bool = True) -> dict:
    """
    load the threshold related data from a file th.txt for a specific layer
    Args:
        calib_load_path [str] a path to a directory which contains th.txt, 
            containing the calibrated thresholds. These thresholds will be 
            loaded.
        layer [int] id of the attention layer for which the threholds should 
            be loaded.
        num_heads [int] number of attention heads in the given layer (each) 
            attention head is expected to have a set of per-seqlen thresholds
        copy_to [str, optional] path to an existing directory of to where 
            the th.txt should be copied from calib_load_path. If not specified, 
            no copy will be created.
    Returns:
        dictionary with the fileds:
        'th_list' : list of per-head dictionaries (each dictionary contains the 
                    per-seqlen threhold value)
        'K'       : integer, the k for which the thresholds were calibrated
        'th_num_samples' : list of per-head dictionaries (each dictionary 
                    contains the per-seqlen numbr of calibration samples)
        'th_fit_params': parameterized thresholds (future feature)
    """
    loaded_th_data = dict(th_list=[{} for _ in range(num_heads)],
                          th_num_samples=[{} for _ in range(num_heads)])
    k_set = set()
    with open(calib_load_path+"/th.txt", "r") as f:
        for line in f.readlines():
            # one line looks like:
            # L31_H31:1219 0.0002257227897644043 4 64
            header, th, num_samples, k = line.split(" ")
            if header.startswith(f"L{layer}_H"):
                head = int(header.split(":")[0].split("_H")[1])
                seqlen= int(header.split(":")[1])
                loaded_th_data['th_list'][head][seqlen] = float(th)
                loaded_th_data['th_num_samples'][head][seqlen] = int(num_samples)
                k_set.add(int(k))
                
            # copy the used thresholds into the current products directory
            if copy_to is not None and layer == 0:
                with open(copy_to+"/th.txt", "a") as f:
                    f.write(line)

    assert len(k_set) == 1, "currently only single k per layer is supported"
    loaded_th_data['K'] = k_set.pop()
    # TODO: add support for loading the parameterized thresholds
    loaded_th_data['th_fit_params'] = None
    
    if verbose:
        print(f"--Calibration (threshold) loaded for layer-{layer}")

    return loaded_th_data

def load_sdc_from_directory(calib_load_path: str, layer:int, num_heads:int, copy_to:str = None, verbose:bool = True) -> dict:
    """
    load the sdc-related data from a file sdc.txt for a specific layer
    Args:
        calib_load_path [str] a path to a directory which contains sdc.txt, 
            containing the calibrated sdc-values. These values will be loaded.
        layer [int] id of the attention layer for which the sdc-values should 
            be loaded.
        num_heads [int] number of attention heads in the given layer (each) 
            attention head is expected to have a set of per-seqlen sdc-values.
        copy_to [str, optional] path to an existing directory of to where 
            the sdc.txt should be copied from calib_load_path. If not specified, 
            no copy will be created.
    Returns:
        dictionary with the fileds:
        'sdc_list' : list of per-head dictionaries (each dictionary contains 
                     the per-seqlen threhold value)
        'sdc_num_samples' : list of per-head dictionaries (each dictionary 
                    contains the per-seqlen numbr of calibration samples)
    """
    loaded_sdc = dict(sdc_list=[{} for _ in range(num_heads)],
                      sdc_num_samples=[{} for _ in range(num_heads)]) 
    with open(calib_load_path+"/sdc.txt", "r") as f:
        for line in f.readlines():
            # one line looks like:
            # L31_H31:1219 0.0002257227897644043 4
            header, sdc, num_samples = line.split(" ")
            if header.startswith(f"L{layer}_H"):
                head = int(header.split(":")[0].split("_H")[1])
                seqlen= int(header.split(":")[1])
                loaded_sdc['sdc_list'][head][seqlen] = float(sdc)
                loaded_sdc['sdc_num_samples'][head][seqlen] = int(num_samples)
            
            # copy the used sdc parameters into the current products directory
            if copy_to is not None and layer == 0:
                with open(copy_to+"/sdc.txt", "a") as f:
                    f.write(line)

    if verbose:    
        print(f"--Calibration (sdc) loaded for layer-{layer}")
        
    return loaded_sdc

def set_params(model, **params):
    for child_name, child in model.named_children():
        if isinstance(child, (LlamaDecoderLayer)):
            attention = child.self_attn
            
            # Set general parameters
            attention.K = params['K']
            attention.calibrate = params['calibrate']
            attention.calibration_phase = params['calibrate']
            attention.mode = params['mode']
            attention.placement = params['placement']
            attention.num_calib_requests = 0
            attention.sdc = params['sdc']
            attention.sdc_scale = params['sdc_scale']
            attention.test_layer = params['test_layer']
            attention.vmc = params['vmc']
            attention.calib_load_path = params['calib_load_path']
            attention.capk = params['capk']
            attention.dump_qkv=params['dump_qkv']
            attention.dump_stats_set={'kept_attn_generative_decoding', 'kept_attn_prefill', 
                                      'kept_vrow_generative_decoding', 'kept_vrow_prefill', 
                                      'kept_vrow_popularities_generative_decoding', 
                                      'kept_vrow_popularities_prefill'} - params['dump_stats_set_exclude']

            # Reset calibraton-related values
            if attention.calib_load_path != "":
                
                # load threhsolds from th.txt (or potentially from a parameterized threshold file)
                if attention.mode == 0:
                    loaded_th_data = load_thresholds_from_directory(attention.calib_load_path, attention.id, attention.num_heads, copy_to=attention.products_dir_path)                    
                    attention.th_list = loaded_th_data['th_list']
                    attention.th_num_samples = loaded_th_data['th_num_samples']
                    attention.th_fit_params = loaded_th_data['th_fit_params']

                # load softmax denominator compensation (sdc) parameters from sdc.txt
                if attention.sdc == "offline-calibrated":
                    loaded_sdc = load_sdc_from_directory(attention.calib_load_path, attention.id, attention.num_heads, copy_to=attention.products_dir_path)
                    attention.sdc_list = loaded_sdc['sdc_list']
                    attention.sdc_num_samples = loaded_sdc['sdc_num_samples']     

            elif params['calibrate']:
                # Calibration requires threshold to be reset and number of samples given as input for calibration
                attention.num_calib_requests = params['calibration_requests']
                attention.obt_calib_requests = 0
                attention.th_list=[{} for _ in range(attention.num_heads)]
                attention.th_num_samples=[{} for _ in range(attention.num_heads)]
                attention.th_fit_params = None
                attention.calib_tac = params['calib_tac']
                attention.calib_add_sigma = params['calib_add_sigma']                
                attention.calib_sample_frac = params['calib_sample_frac']
                if attention.sdc == "offline-calibrated":
                    attention.sdc_list = [{} for _ in range(attention.num_heads)]
                    attention.sdc_num_samples = [{} for _ in range(attention.num_heads)]
        else:
            set_params(child, **params)
            
