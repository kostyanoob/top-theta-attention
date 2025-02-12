import itertools
import pdb
import numpy as np
import torch
from transformers.models.llama.modeling_llama import *
import torch.nn.functional as F
import math


def merge_intervals(list_of_lists_of_intervals: List[List[Tuple[float,float,int]]], drop_singleton_intervals:bool=True, verify_same_ks:bool=False) -> List[Tuple[float,float,int,int]]:
    """
    Gets a list of per-sample lists. Each per-sample list describes the 
    threshold intervals obtained from a single calibration sample(s), and 
    aggregates them all into a single list (summing up the Ks and counting 
    the overlapping intervals)

    Params:
        list_of_lists_of_intervals - a list of lists threshold intervals
            where each element in each (sub)list is a 4-tuple:
            0) threshold_i_start
            1) threshold_i_end
            2) sum Ks of overlapping intervals represented by this interval
            3) num of overlapping intervals represented by this interval
        drop_singleton_intervals - if True then threshold interval in which start_th==end_th will be omitted from the merging.
                                   Dropping the singletons seems to be a good idea because they are mainly 
                                   a result of repetitive elements in the attention vector during calibration.
        verify_same_ks - setting it to True will check that all the lists in <list_of_lists_of_intervals>
            are of the same length.

    Returns:
    Merged list where the elements are 4-tuples with the following fields:
        - threshold_i_start
        - threshold_i_end 
        - sum of ks in all overlapping thresholds
        - number of overlapping thresholds
    """
    if verify_same_ks:
        num_ks = len(list_of_lists_of_intervals[0])
        assert all(len(l)==num_ks for l in list_of_lists_of_intervals)

    # all thresholds that were observed for this (layer,head,seq_len)
    all_thresholds_lo_to_hi = sorted(set(interval[0] for intervals_of_one_sample in list_of_lists_of_intervals for interval in intervals_of_one_sample)) 
    
    # keep track which intervals were covered so far for each of the calibration samples
    sample_pointers = [0 for _ in range(len(list_of_lists_of_intervals))]

    # construct a merged list of intervals, each interval is a 4-tuple: (th_start, th_end, cummulative_k, num_overlapping_thresholds)
    merged_interval_lst = list()

    for th_start, th_end in zip(all_thresholds_lo_to_hi[:-1], all_thresholds_lo_to_hi[1:]):
        cummulative_k = 0
        num_overlapping_thresholds = 0

        # go through all the per-sample interval lists (which are sorted from low to hi threshold )
        for sample_id, sample_interval_lst in enumerate(list_of_lists_of_intervals):
            if sample_pointers[sample_id] == len(sample_interval_lst):
                continue  # all the intervals of this sample have been consumed

            sample_tuple = sample_interval_lst[sample_pointers[sample_id]] 

            # consume all the singleton intervals (start==end) of this calibration sample's list
            while (sample_tuple is not None) and (sample_tuple[0] == sample_tuple[1] == th_start):
                if not drop_singleton_intervals:
                    cummulative_k += sample_tuple[2]
                    num_overlapping_thresholds += sample_tuple[3]
                sample_pointers[sample_id] += 1
                sample_tuple = sample_interval_lst[sample_pointers[sample_id]] if sample_pointers[sample_id] < len(sample_interval_lst) else None
            
            # consume the current non-singleton (start<end) interval if it is excatly the interval [th_start,th_end)]
            if sample_tuple is not None and sample_tuple[0] <= th_start and th_end <= sample_tuple[1]:
                cummulative_k += sample_tuple[2]
                num_overlapping_thresholds += sample_tuple[3]
                if sample_tuple[1] == th_end:
                    sample_pointers[sample_id] += 1
                    
        merged_interval_lst.append((th_start, th_end, cummulative_k, num_overlapping_thresholds))

    return merged_interval_lst

def threshold_from_merged_intervals(merged_intervals: List[Tuple[float,float,int,int]], K: int) -> Tuple[float, float]:
    """
    Gets as an input the merged list where the elements are 
    (threshold_i_start, threshold_i_end, sum of effective ks in all overlapping thresholds, number of overlapping thresholds) 

    mean_effective_K(inteval) = sum_effective_ks/number_overlapping_thresholds
    
    Returns a threshold that gives the closest center of an interval that, and the mean effective K for it
    """
    assert len(merged_intervals) > 0
    min_abs_dist_from_K = float('inf')
    opt_threshold = float('inf')
    opt_mean_effective_K = 0
    for th_start, th_end, sum_effective_ks, number_overlapping_thresholds in merged_intervals:
        center_th = (th_end + th_start) / 2 if th_end < float('inf') else th_start
        mean_effective_K = sum_effective_ks / number_overlapping_thresholds
        abs_dist_from_K = abs(mean_effective_K - K)
        if abs_dist_from_K < min_abs_dist_from_K:
            opt_threshold = center_th
            min_abs_dist_from_K = abs_dist_from_K
            opt_mean_effective_K = mean_effective_K
    return opt_threshold, opt_mean_effective_K

class TopK_LLamaAttention(LlamaAttention):
    def __init__(self, config : LlamaConfig, id: int, reduce_gpu_mem: bool = False, products_dir_path: str = "products"):
        super().__init__(config)
        self.to(config.torch_dtype) # Note in theory the super class should instantiate the module in torch_dtype. Might be obsolote for newer versions of transformers

        self.K = -1            # K value
        self.id = id           # Layer id
        self.calibrate = False # To put in calibration mode
        self.mode = 3          # 0-TH, 1-TOPK, other than 0/1 for baseline
        self.placement = 'none' # 'pre-sofmtax' or 'post-softmax' - for topk/th; 'none' - for baseline

        self.num_calib_requests=0  # total number of requests to use for calibration
        self.obt_calib_requests=0  # current number of processed calibration requests      
        self.calib_tac=False       # topk-at-calibration (applies only for top-th)
        self.th_list=[]            # final th list vs seqlen {head_num: {LEN : TH}}
        self.th_num_samples=[]     # num samples for every seq len in calibration {head_num: {LEN : num_samples_LEN}}
        self.th_fit_params=None    # not used
        self.k_sweep_full_lst=None # longest list of ks to sample thresholds for, during calibration. In practise, for calibration samples with  sequence length <  this list length: only a prefix of this list is used
        self.test_layer = None     # Layer to be tested, None -> all layers tested

        self.rng = np.random.default_rng(42)
        self.reduce_gpu_mem = reduce_gpu_mem
        self.products_dir_path = products_dir_path  # per-layer thresholds from the calibration are written here

        self.vmc = False           # v-mean compensation (applies only for top-k/th)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        # padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        '''if self.id==0:
            logging.info(f'hidden states:{hidden_states.size()}')
            logging.info(f'attention_mask: {attention_mask.size()}')
            logging.info(f'position_ids: {position_ids.size()}')
            if past_key_value!= None:
                logging.info(f'past_key_value: {past_key_value[0].size()}  {past_key_value[1].size()}')
            logging.info(f'output_attention: {output_attentions}')
            logging.info(f'use_cache: {use_cache}')'''
            
        bsz, q_len, _ = hidden_states.size()


        # assert self.config.pretraining_tp <= 1, 'different to llama attention'
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # q = [bsz, q_len, hidden_size]         
        # k = [bsz, q_len, hidden_size/num_key_value_groups]      
        # v = [bsz, q_len, hidden_size/num_key_value_groups]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        # q = [bsz, num_heads, q_len, head_dim]         
        # k = [bsz, num_key_value_heads, q_len, head_dim]      
        # v = [bsz, num_key_value_heads, q_len, head_dim]

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Grouped-Query-Attention (GQA)
        # Repeat the k, v states according to the number of groups (so that dimensions of k, v will *match* the one of Q)
        # This will have no effect when self.num_key_value_groups == 1 (i.e. no grouped query attention is applied)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # k = [bsz, num_heads=num_key_value_groups*num_key_value_heads, q_len, head_dim]      
        # v = [bsz, num_heads=num_key_value_groups*num_key_value_heads, q_len, head_dim]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
        
        #-----------------------------Top-K * TH Implementation ----------------------------------------
        if self.placement == 'pre-softmax':
            max_attention_scores = attn_weights.max(dim=-1)[0].to(dtype=torch.float32) if self.sdc != 'none' and self.sdc_scale > 0.0 else None  # DO THIS BEFORE SOFTMAX and only if SDC compensation is necessary, otherwise skip this computation
            attn_weights, attn_scores_unselected, attn_top_mask = self.topk_or_threshold(attn_weights, query_states, kv_seq_len)
            existing_denoms = (attn_weights.to(dtype=torch.float32) - max_attention_scores.unsqueeze(-1)).exp().sum(dim=-1) if max_attention_scores is not None else None  # DO THIS BEFORE SOFTMAX compute the per-row denominators of the softmax, considering only the top (selected) elements. Do this step only if SDC compensation is necessary, otherwise skip this computation
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = self.softmax_denominator_compensation(attn_weights, attn_scores_unselected, attn_top_mask, existing_denoms, max_attention_scores, dtype=torch.float32).to(query_states.dtype)
        elif self.placement == 'post-softmax':
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights, _, _ = self.topk_or_threshold(attn_weights, query_states, kv_seq_len)
        elif self.placement == 'none': 
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)     
        else: 
            raise ValueError(f'Illegal topk placement encountered: "{self.placement}"')
        #-----------------------------------------------------------------------------------------
        attn_output = torch.matmul(attn_weights, value_states)

        if self.vmc:
            attn_output = self.v_mean_compensation(attn_output, attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def per_k_th_lst_to_per_k_interval_lst(self, th_lst: List[float], count_component:Union[None,float,int] = None) -> List[Tuple[float,float,int]]:
        """
        Gets a list of thresholds that correspond to the list of k=[0, 1,...,m-1].
        the "m" is the number of k's for which these thresholds were calibrated. and it can vary from one list to another.
        Returns a list of weighted intervals of length "m", where each elements is one of the two options
         3-tuple: (threshold_i_start, threshold_i_end, ki) - if count_component is None
         4-tuple: (threshold_i_start, threshold_i_end, ki, count_component) - if count_component is not None

        Pre-conditions: the th_lst contains monotonically decreasing thresholds, 
        because they must correspond to monotonically increasing ks=[0,1,...,m]      

        Post-conditions: 
            1st interval is defined as (threshold_m-1_start, threshold_m-2_start, k=m-1)
            i^th (i=0,1,2,3..,m-1) interval is defined as (threshold_i_start, threshold_i-1_start, k=m-1-i)
            m-1^th interval is defined as (threshold_0_start, +inf, k=0)

        It is important to note that each interval is half-open - 
        that is it includes the start and excludes the end.
        """
        # the inputs are ordered according to (decreasing th) <--> (increasing k)
        threhold_starts = th_lst
        threshold_endings = [float('inf')] + th_lst[:-1]
        k_lst = self.k_sweep_full_lst[:len(th_lst)]
        
        # re-order the intervals according to (increasing th) <--> (decreasing k)
        if count_component is None:
            # each interval tuple is a 3-tuple
            per_k_interval_lst = list(zip(reversed(threhold_starts), 
                                          reversed(threshold_endings), 
                                          reversed(k_lst)))
        else:
            # each interval tuple is a 4-tuple: count component as the 4th element
            per_k_interval_lst = list(zip(reversed(threhold_starts), 
                                          reversed(threshold_endings), 
                                          reversed(k_lst),
                                          (count_component for _ in range(len(k_lst)))))            
        assert all(intrvl[0] <= intrvl[1] for intrvl in per_k_interval_lst) # verify that the thresholds are monotonically non-increasing (should be strictly decreasing, but are not due to numerical precision of the threshold)
        assert all(intrvl_hi[2] > intrvl_lo[2] for intrvl_hi, intrvl_lo in zip(per_k_interval_lst[:-1], per_k_interval_lst[1:])) # verify that the ks are monotonically increasing
        return per_k_interval_lst

    def get_threshold(self, head_id, seq_len: int, normalize=False) -> float:
        """
        look-up the threshold of the attention head number <head_id> associated
        with the closest sequence length from calibrated set. 

        normalize=True normalizes the threshold by the numer of samples (use it 
        when thresholds are needed during the calibration and therefore are still 
        not normalized)
        """
        threshold = self.th_list[head_id].get(seq_len, None)

        if threshold is None:
            closest_seq_len = min(self.th_list[head_id].keys(), key = lambda key: abs(key-seq_len))
            threshold = self.th_list[head_id][closest_seq_len]
            if normalize:
                threshold = threshold / self.th_num_samples[head_id][closest_seq_len]

        return threshold

    def sample_rowids(self, row_sample_fraction:float, seq_len, k) -> List[float]:
        """
        Returns chosen at random list of row indices in [0,seq_len - k) 
        The number of indices is equal to row_sample_fraction * seq_len
        """
        sample_population = list(range(0, seq_len - k))
        if 0 < row_sample_fraction < 1.0:
            row_sample_size = math.ceil(len(sample_population) * 0.1)
            row_obtained_counts = [self.th_num_samples[0].get(l + k + 1, 0) for l in sample_population]
            row_desired_counts = max(row_obtained_counts) + 1 - np.array(row_obtained_counts)
            row_sample_prob = row_desired_counts / row_desired_counts.sum() if row_desired_counts.sum() > 0 else np.ones_like(row_desired_counts)/len(row_desired_counts)
            sampled_row_th_rowids = self.rng.choice(sample_population, size=row_sample_size, replace=False, p=row_sample_prob)  # in [0,seq_len-K)
        elif row_sample_fraction == 1.0:
            sampled_row_th_rowids = sample_population
        else: 
            assert False, "Bad fraction. Must be in (0,1.0]"
        
        return sampled_row_th_rowids

    def get_threshold_tensor(self, head_start, head_end, seq_len_start, seq_len_end, normalize=False) -> torch.Tensor:
        """
        returns a 2D tensor th of the shape:
        [head_end-head_start+1, row_end-row_start+1] 
        filled with thresholds, where th[head,seq_len] will contain the floating
        point threshold corresponding to <head> and attention row <seq_len-1>
        """
        return torch.Tensor([[self.get_threshold(head_id, seq_len, normalize) for seq_len in range(seq_len_start,seq_len_end+1)] for head_id in range(head_start, head_end+1)])

    def v_mean_compensation(self, 
                            attn_output: torch.Tensor, 
                            attn_weights: torch.Tensor, 
                            value_states: torch.Tensor) -> torch.Tensor:
        """
        Apply V-mean compensation on the attn_output matrix. This compensation
        affects only when the attn_weights has rows (3rd dimension) that sum up
        to < 1.0 and therefore the attn_weights*V product misses some of the V-rows.
        This compensation is aimed to approximately add these missing V-rows back.

        Args:
            attn_output:  product of the softmax output (attn_weights)
                          multiplied by the V matrix. 
                          shape: (BSZ, NHEADS, SEQ_LEN, HEAD_DIM)
            attn_weights: tensor containing the softmax output, aka attention
                          scores, aka attention probabilities.
                          shape: (BSZ, NHEADS, SEQ_LEN, SEQ_LEN)
            value_states: the value matrix
                          shape: (BSZ, NHEADS, SEQ_LEN, HEAD_DIM)
        
        Returns:
          attention output tensor of the same shape as before, but with every row 
          added a special HEAD_DIM-long compensation vector
        """
        BSZ, NHEADS, SEQ_LEN, HEAD_DIM = attn_output.shape
        preserved_probability_mass = attn_weights.sum(dim=-1)  # sum up each row -> [BSZ, NHEADS, SEQ_LEN]
        lost_probability_mass = 1 - preserved_probability_mass  # take the complementary to represent the probability mass that was removed by the sparsification (topk/th) -> [BSZ, NHEADS, SEQ_LEN]
        v_mean_rows = value_states.cumsum(dim=2) / torch.arange(start=1, end=SEQ_LEN + 1, step=1, device=value_states.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # for every row r in [0,SEQ_LEN-1] compute its causal running v_mean[r,:] vector as an average across V[0:r,:].mean(dim=0) -- > [BSZ, NHEADS, SEQ_LEN, HEAD_DIM]
        attn_output = attn_output + lost_probability_mass.unsqueeze(-1) * v_mean_rows  # [BSZ, NHEADS, SEQ_LEN, HEAD_DIM] + [BSZ, NHEADS, SEQ_LEN, unsqueezed] * [BSZ, NHEADS, SEQ_LEN, HEAD_DIM]
        return attn_output

    def softmax_denominator_compensation(self, 
                                         attn_weights: torch.Tensor, 
                                         attn_scores_unselected: torch.Tensor, 
                                         attn_top_mask: torch.Tensor,
                                         existing_denoms: torch.Tensor,
                                         max_attention_scores: torch.Tensor,
                                         dtype=torch.float32
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
          attn_weights - attention tensor (after the softmax was applied)
                         Shape: (BATCH_SIZE, NHEADS, SEQ_LEN, SEQ_LEN)
          attn_scores_unselected - attention tensor (before the softmax was applied)
                         with -inf in all the places which 
                         were chosen to be kept by topk/thinfinity
          attn_top_mask - boolean mask with True in places that were selected 
                         to keep.
                         or topk. Shape: (BATCH_SIZE, NHEADS, SEQ_LEN)
          existing_denoms - sum of e^{xi-max(x)} (before softmax) per every row
                         of the attetnion matrix *after-top-k/th* but *before-softmax*
                         Shape: (BATCH_SIZE, NHEADS, SEQ_LEN)
          max_attention_scores -  per attention row (per head ber batch dim)
                         maximum of the elements (must be computed before the 
                         softmax. i.e. on the attention scores, aka logits) 
                         Shape: (BATCH_SIZE, NHEADS, SEQ_LEN)
          dtype -        data type, in which to process the compensation
        Returns:
          attention matrix of the same shape as before, but with the attention 
          scores renormalized by a larger denominator 
        """
        
        BATCH_SIZE, NHEADS, SEQ_LEN, _ = attn_weights.size()
        K = self.K[self.id] 

        # no compensation required
        if self.sdc == 'none' or self.sdc_scale == 0.0 or SEQ_LEN <= K or attn_top_mask is None:
            return attn_weights

        if self.sdc == 'exact':
            assert ((BATCH_SIZE, NHEADS, SEQ_LEN, SEQ_LEN) == attn_scores_unselected.size())
            assert ((BATCH_SIZE, NHEADS, SEQ_LEN) == existing_denoms.size())
            assert ((BATCH_SIZE, NHEADS, SEQ_LEN) == max_attention_scores.size())
            missing_denominator_term = torch.sum(torch.exp(attn_scores_unselected[:,:,K:,:].to(dtype=dtype) - max_attention_scores[:,:,K:].unsqueeze(-1).to(dtype=dtype)), dim=-1)
            new_denom = existing_denoms[:,:,K:] + self.sdc_scale * missing_denominator_term
            attn_weights[:,:,K:,:] = attn_weights[:,:,K:,:].mul(existing_denoms[:,:,K:].unsqueeze(-1)).div(new_denom.unsqueeze(-1))
        elif self.sdc == 'exp-threshold':
            assert self.mode == 0, "exp-threshold compensation is only applicable in mode=0 (thresholding)"
            if not self.calibrate:
                assert ((BATCH_SIZE, NHEADS, SEQ_LEN, SEQ_LEN) == attn_top_mask.size())
                assert ((BATCH_SIZE, NHEADS, SEQ_LEN) == max_attention_scores.size())

                num_selected_elements = attn_top_mask.sum(dim=-1)
                num_unselected_elements = (SEQ_LEN - num_selected_elements)
                
                # look-up the SEQ_LEN-K thresholds from the closest sequence length from calibrated set. 
                per_head_row_thresholds = self.get_threshold_tensor(0, NHEADS - 1, K + 1, SEQ_LEN)  # [NHEADS, SEQ_LEN-K] - a 2D matrix of per-head-per-row thresholds 
                per_head_row_thresholds = per_head_row_thresholds.to(device=attn_weights.device, dtype=attn_weights.dtype)

                # Renormalize softmax score (e^a_i/existing_denoms[row]) by multiplying
                # it by "existing_denoms[row]/(existing_denoms[row] + sdc_scale * num_unselected[row] * e^(th - max(a))"
                missing_denominator_term = num_unselected_elements[:,:,K:] * torch.exp(per_head_row_thresholds.unsqueeze(0) - max_attention_scores[:,:,K:].to(dtype=existing_denoms.dtype))
                new_denom = existing_denoms[:,:,K:] + self.sdc_scale * missing_denominator_term
                attn_weights[:,:,K:,:] = attn_weights[:,:,K:,:].mul(existing_denoms[:,:,K:].unsqueeze(-1)).div(new_denom.unsqueeze(-1))

        elif self.sdc == 'offline-calibrated':
            assert self.mode in {0,1}, "offline-calibrated compensation is available in mode 0 (thresholding) and 1 (topk)"
            if not self.calibrate:

                # look-up per-row missing denominator terms from the calibrated dictionary
                missing_denominator_terms = torch.Tensor([self.sdc_list.get(seq_len) or self.sdc_list[min(self.sdc_list.keys(), key = lambda key: abs(key-seq_len))] for seq_len in range(K+1,SEQ_LEN+1)])
                missing_denominator_terms = missing_denominator_terms.to(dtype=attn_weights.dtype, device=attn_weights.device)

                # Renormalize softmax score of row vector a "e^(a_i-max(a))/existing_denoms[row]" by multiplying
                # it by "existing_denoms[row]/(existing_denoms[row] + sdc_scale * missing_denominator_term_from_calibration[row])"
                new_denom = existing_denoms[:,:,K:] + self.sdc_scale * missing_denominator_terms.unsqueeze(0).unsqueeze(0)
                attn_weights[:,:,K:,:]  = attn_weights[:,:,K:,:] .mul(existing_denoms[:,:,K:].unsqueeze(-1)).div(new_denom.unsqueeze(-1))

        else:
            assert NotImplementedError(f"self.sdc compensation is not supported")

        return attn_weights

    def topk_or_threshold(
            self,
            attn_weights: torch.Tensor,
            query_states: torch.Tensor, 
            kv_seq_len: int,
        ) -> torch.Tensor:
        """
        apply the top-k or the thresholding (according to the self.mode) on 
        the attn_weights tensor across its last dimension. This results in 
        keeping only a limited number of elements within each row of each head's
        attention weight matrix)

        Arguments
            attn_weights - Tensor of shape (bsz, self.num_heads, q_len, kv_seq_len).
                           This is the primary input tensor to be processed.
            query_states - Tensor of shape (bsz, self.num_heads, q_len, self.head_dim).
                           Used only for shape and dtype information.
            kv_seq_len   - Integer, normally equal to sequence length, used
                           for logging.

        Returns: 3 tensors:
                 1) the attn_weights tensor (bsz, self.num_heads, q_len, kv_seq_len) 
                 after the application of Top-k / thresholding.
                 2) attn_weights_unselected - the complementary tensor to attn_weights,
                    where the not selected weights are equal to original weight, whereas the
                    selected weights are replaced by 0 or -inf. Will be None if self.sdc is 'none'
                 3) attn_top_mask - boolean mask signifying the elements that were selected
                 to be kept by topk/th. Will be None if self.sdc is 'none'
        """
        BATCH_SIZE, NHEADS, SEQ_LEN, DIM = query_states.size()
        K = self.K[self.id] if isinstance(self.K, list) else self.K
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
                # ----- Log Sequence Lengths -----
                # logging.info(f'ALL SIZES:{query_states.size()} {key_states.size()} {value_states.size()}')
                # logging.info(f'{query_states.dtype} {key_states.dtype} {value_states.dtype}')
                if query_states.size()[2] != 1:
                    with open(f"{self.products_dir_path}/sequence_lengths_per_example.csv",'a') as f:
                        f.write(f'{kv_seq_len}\n')                    
            
            # ----- mode-0 Thresholding -----
            if self.mode == 0:
                if (self.id == self.test_layer) or (self.test_layer == None):
                    attn_top_mask = torch.full(attn_weights.size(), True, dtype=torch.bool, device=attn_weights.device).tril() # re-enforces causality
                    attn_scores_unselected = torch.full_like(attn_weights, replacement_value) if self.sdc != 'none' and self.sdc_scale > 0.0 else None  # cancel allocation when no further use of this tensor will be made 
                    if 0 < K < SEQ_LEN:
                        per_head_row_thresholds = self.get_threshold_tensor(0, NHEADS - 1, K + 1, SEQ_LEN)  # [NHEADS, SEQ_LEN-K] - a 2D matrix of per-head-per-row thresholds 
                        per_head_row_thresholds = per_head_row_thresholds.to(device=attn_weights.device, dtype=attn_weights.dtype)
                        attn_top_mask[:,:,K:,:] = torch.gt(attn_weights[:,:,K:,:], per_head_row_thresholds.unsqueeze(0).unsqueeze(-1))
                        if attn_scores_unselected is not None:
                            attn_scores_unselected[:,:,K:,:] = torch.where(attn_top_mask[:,:,K:,:], replacement_value, attn_weights[:,:,K:,:])
                        attn_weights[:,:,K:,:] = torch.where(attn_top_mask[:,:,K:,:], attn_weights[:,:,K:,:], replacement_value)
                        buff_occupancy_topk_per_head = BATCH_SIZE * (((1 + K) * K / 2) + ((SEQ_LEN - K) * K))  # number of attention elements that the topk method would keep per attention head
                    else:
                        buff_occupancy_topk_per_head = BATCH_SIZE * ((1 + SEQ_LEN) * SEQ_LEN / 2)  # number of attention elements that the topk method would keep per attention head
                    
                    # Collect relative (to topk method) number of attention 
                    # elements that survived the thresholding.
                    # buff_occupancy_relative should be ~ 1 for good thresholding.
                    buff_occupancy_relative_per_head = attn_top_mask.sum(dim=(2,3)) / buff_occupancy_topk_per_head
                    with open(f"{self.products_dir_path}/layer{self.id}.txt",'a') as f:
                        for b_, h_ in itertools.product(range(BATCH_SIZE), range(NHEADS)):
                            f.write(f'L{self.id}_H{h_}:{SEQ_LEN} {K} per-head-row-th {buff_occupancy_relative_per_head[b_, h_]}\n')
                    
                    if attn_scores_unselected is None:
                        attn_top_mask = None
            
            # ----- mode-1 TopK -----               
            if self.mode == 1:
                if (self.id == self.test_layer) or (self.test_layer == None):
                    attn_top_mask = torch.full(attn_weights.size(), True, dtype=torch.bool, device=attn_weights.device).tril() # re-enforces causality
                    attn_scores_unselected = torch.full_like(attn_weights, replacement_value) if self.sdc != 'none' and self.sdc_scale > 0.0 else None  # cancel allocation when no further use of this tensor will be made 
                    if 0 < K < SEQ_LEN:
                        # Find Top-k elements per row in attention rows [K, K+1,...]:
                        vals, idxs = attn_weights[:,:,K:,:].topk(K, dim=-1)
                        attn_top_mask[:,:,K:,:].fill_(False).scatter_(-1, idxs, True)   
                        if attn_scores_unselected is not None:
                            attn_scores_unselected[:,:,K:,:] = torch.where(attn_top_mask[:,:,K:,:], replacement_value, attn_weights[:,:,K:,:])
                        attn_weights[:,:,K:,:] = torch.where(attn_top_mask[:,:,K:,:], attn_weights[:,:,K:,:], replacement_value)
                    
                    if attn_scores_unselected is None:
                        attn_top_mask = None

        else:
            # ----- Performing Calibration -----
            assert(self.num_calib_requests != 0), f"number of calibration requests was not set"
            if self.mode == 0 and 0 < K < SEQ_LEN:
                # -- calibration of thresholds for top-th --
                # Thresholding based on Top-K:

                # Step 1 - find multiple thresholds: for a list of ks
                q_vec = torch.Tensor([(1 - k / SEQ_LEN) for k in self.k_sweep_full_lst if k < SEQ_LEN]).to(device=attn_weights.device)
                if self.reduce_gpu_mem:
                    # chunked (several heads per chunk) quantile computation on gpu to keep the memory requirements low
                    quant_chunks = []
                    for attn_heads_chunk in torch.tensor_split(attn_weights[:,:,K:,:], 4, dim=1):
                        quant_chunk = torch.quantile(attn_heads_chunk.float(), q_vec , dim=3, interpolation='lower')  # quantile() requires the input tensor dtype to be either float or double
                        quant_chunks.append(quant_chunk)
                    quant = torch.cat(quant_chunks, dim=1) 
                else:
                    quant = torch.quantile(attn_weights[:,:,K:,:].float(), q_vec , dim=3, interpolation='lower')  
                # quant tensor (len(q_vec), NUM_BATCH_SIZE, NHEADS, SEQ_LEN)
                # contains a threshold per row of attention matrix (per head per batch example) 

                # Step 2 - record the thresholds
                # sample 10% of rows to actually calibrate on (bias towards the less sampled ones so far)
                sampled_row_th_rowids = self.sample_rowids(0.1, SEQ_LEN, K)  # row indices in ~ U[0, seq_len - k) 
                # log_row_head_th = {row_seq_len + K + 1: [] for row_seq_len in sampled_row_th_rowids}  # DELETE ME IF TH_LOG.TXT is NOT NEEDED
                for sample_id, head_id, quant_row_id in itertools.product(range(BATCH_SIZE), range(NHEADS), sampled_row_th_rowids):
                    rowid_attn = quant_row_id + K  # attention row corresponding to this quant row
                    row_seq_len = rowid_attn + 1  # the sequence length corresponding to this threshold row
                    per_k_th_lst = quant[:min(len(q_vec), row_seq_len), sample_id, head_id, quant_row_id].tolist() # list containing row_seqlen-1 thresholds

                    # convert this calibraiton sample's list of per-k *thresholds* into a list of per-k *intervals* 
                    # each interval will include its start and exclude its end ([q_i, q_(i+1)), k_i, 1] and associated with a specific effective k:
                    list_of_intervals = self.per_k_th_lst_to_per_k_interval_lst(per_k_th_lst, count_component=1)
                    
                    # merge the list of threshold intervals into a single unified list. Each element is [q_i, q_(i+1)), sum of effective ks in the overlapping intervals so far, num of overlapping intervals so far]
                    # log_row_head_th[row_seq_len].append(per_k_th_lst)  # DELETE ME IF TH_LOG.TXT is NOT NEEDED
                    if row_seq_len not in self.th_list[head_id]:
                        self.th_list[head_id][row_seq_len] = list_of_intervals
                        self.th_num_samples[head_id][row_seq_len] = 1
                    else:
                        # merge 2 lists of threshold intervals: (accumulated so far list, new list)
                        self.th_list[head_id][row_seq_len] = merge_intervals([self.th_list[head_id][row_seq_len], 
                                                                            list_of_intervals],
                                                                            drop_singleton_intervals=False) 
                        self.th_num_samples[head_id][row_seq_len] += 1

                # # DELETE ME IF TH_LOG.TXT is NOT NEEDED
                # with open(f"{self.products_dir_path}/th_log.txt",'a') as f:
                #     for head_id, sampled_row_th_rowids in itertools.product(range(NHEADS), sampled_row_th_rowids):
                #         rowid_attn = quant_row_id + K 
                #         row_seq_len = rowid_attn + 1
                #         row_thresholds_of_this_head_for_all_sweep_ks = ",".join(map(str, log_row_head_th[row_seq_len][head_id]))
                #         f.write(f'L{self.id}_H{head_id} {K} {row_seq_len} {row_thresholds_of_this_head_for_all_sweep_ks}\n')

            if self.sdc == 'offline-calibrated' and 0 < K < SEQ_LEN:
                # -- calibration for softmax denominator compensation --
                assert replacement_value == torch.finfo(attn_weights.dtype).min  # implicitly validates pre-softmax placement
                if self.mode == 0:
                    #  apply the opposite of the top-th (keep below-or-equal to threshold)- with the so-far calibrated threshold 
                    per_head_row_thresholds = self.get_threshold_tensor(0, NHEADS - 1, K + 1, SEQ_LEN, normalize=True)  # [NHEADS, SEQ_LEN-K] - a 2D matrix of per-head-per-row thresholds. Also normalize! because the thresholds are now only sums (calibration ongoing)
                    per_head_row_thresholds = per_head_row_thresholds.to(device=attn_weights.device, dtype=attn_weights.dtype)
                    unselected_attn_weights = torch.where(attn_weights[:,:,K:,:] <= per_head_row_thresholds.unsqueeze(0).unsqueeze(-1), 
                                                          attn_weights[:,:,K:,:], 
                                                          replacement_value).to(torch.float32)

                elif self.mode == 1:
                    #  apply bottom-(N-k), check how many elements left per row                   
                    vals, idxs = attn_weights[:,:,K:,:].topk(SEQ_LEN-K, dim=-1, largest=False, sorted=False) # Non-top-k <==> Bottom-N-K
                    unselected_attn_weights = torch.full(attn_weights[:,:,K:,:].size(), 
                                                         replacement_value, 
                                                         dtype=attn_weights.dtype,
                                                         device=attn_weights.device).scatter_(-1, idxs, vals).to(torch.float32)
                else:
                    assert False, "--sdc 'offline-calibrated' is only allowed for mode=0 or 1."
                
                # find sum(exp(a_i - max(a))) across each row vecotr "a", where a_i are non-top-k / below-threshold elements. Note that the max(a) s taken across all the elements (including the kept once)
                per_row_exp_maxes = attn_weights[:,:,K:,:].max(dim=-1, keepdims=True)[0]
                per_row_exp_sums = torch.exp(unselected_attn_weights - per_row_exp_maxes).sum(dim=-1).to(query_states.dtype)  # [BATCH_SIZE, NHEADS, SEQ_LEN]
                per_row_avg_exp_sum = per_row_exp_sums.view([BATCH_SIZE * NHEADS, SEQ_LEN - K]).mean(0)  # for every token position (attenton row) find an average
                for row, avg_exp_sum in enumerate(per_row_avg_exp_sum):
                    row_seq_len = row + K + 1
                    if row_seq_len not in self.sdc_list:
                        self.sdc_list[row_seq_len] = avg_exp_sum.item()
                        self.sdc_num_samples[row_seq_len] = 1
                    else:
                        self.sdc_list[row_seq_len] += avg_exp_sum.item()
                        self.sdc_num_samples[row_seq_len] += 1

            if self.calib_tac and self.mode == 0 and 0 < K < SEQ_LEN:
                # process the attn_weights as if top-k was performed. This 
                # should help the subsequence layers of the model to calibrate 
                # on a more accurately represented (sparsified) activations
                vals, idxs = attn_weights[:,:,K:,:].topk(K, dim=-1)
                attn_weights[:,:,K:,:] = torch.full(attn_weights[:,:,K:,:].size(), 
                                                    replacement_value, 
                                                    dtype=attn_weights.dtype, 
                                                    device=attn_weights.device).scatter_(-1, idxs, vals)

            self.obt_calib_requests += 1
            if self.obt_calib_requests == self.num_calib_requests:
                if self.mode == 0:
                    # for every observed sequence length - set the threshold to be the one that on average (across the calibration set) provides number of elements closest to
                    mean_eff_K_dict = {}
                    with open(f"{self.products_dir_path}/th_effectiveK.txt",'a') as f:
                        for head_id in range(NHEADS):
                            seq_len_to_th_dict = {}
                            mean_eff_K_dict[head_id] = {}
                            for seqlen, merged_intervals in sorted(self.th_list[head_id].items()):
                                # select the threshold that gives the desired effective K, from the merged list of thresholds
                                seq_len_to_th_dict[seqlen], mean_eff_K_dict[head_id][seqlen] = threshold_from_merged_intervals(merged_intervals, K)

                                # dump to a text file the list of (th, effective-K)
                                th_mean_eff_K_lst = [f"{(t[1] + t[0])/2}>{t[2]/t[3]}" for t in merged_intervals]
                                th_mean_eff_K_str = ",".join(th_mean_eff_K_lst)
                                f.write(f'L{self.id}_H{head_id}:{seqlen} th>mean_eff_K {th_mean_eff_K_str}\n')

                            # keep the threshold we've selected instead of the lists of lists 
                            # threshold lists with a single floating-point threhsold per sequence length
                            self.th_list[head_id] = seq_len_to_th_dict

                    with open(f"{self.products_dir_path}/th.txt",'a') as f:
                        for head_id in range(self.num_heads):
                            for seqlen, th in self.th_list[head_id].items():
                                f.write(f'L{self.id}_H{head_id}:{seqlen} {th} {self.th_num_samples[head_id][seqlen]} {mean_eff_K_dict[head_id][seqlen]}\n')
                    print(f"--Calibration (threshold) done for layer-{self.id}")

                if self.sdc == 'offline-calibrated':
                    # TODO: remove this method or make it per-head adapted.
                    # for every observed sequence length - the compensation term
                    #  to keep is the average across calibration samples
                    print(f"--Calibration (sdc) done for layer-{self.id}")
                    self.sdc_list = { seqlen : sdc / self.sdc_num_samples[seqlen] for seqlen, sdc in sorted(self.sdc_list.items()) }
                    with open(f"{self.products_dir_path}/sdc.txt",'a') as f:
                        for seqlen, sdc in self.sdc_list.items():
                            f.write(f'{seqlen} {sdc}\n')                    

        return attn_weights, attn_scores_unselected, attn_top_mask

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
            
def set_params(model, **params):
    for child_name, child in model.named_children():
        if isinstance(child, (LlamaDecoderLayer)):
            attention = child.self_attn
            
            # Modify K and Threshold
            attention.K = params['K']
            attention.calibrate = params['calibrate']
            attention.mode = params['mode']
            attention.placement = params['placement']
            attention.num_calib_requests = 0
            attention.sdc = params['sdc']
            attention.sdc_scale = params['sdc_scale']
            attention.test_layer = params['test_layer']
            attention.vmc = params['vmc']

            # Reset stat values
            if params['calibrate']:
                # Calibration requires threshold to be reset and number of samples given as input for calibration
                attention.num_calib_requests = params['calibration_requests']
                attention.obt_calib_requests = 0
                attention.th_list=[{} for _ in range(attention.num_heads)]
                attention.th_num_samples=[{} for _ in range(attention.num_heads)]
                attention.th_fit_params = None
                attention.k_sweep_full_lst = list(range(0, 5000))     #list(range(0, 2 * attention.K[attention.id]))
                if attention.sdc == "offline-calibrated":
                    attention.sdc_list = {}
                    attention.sdc_num_samples = {}       
                attention.calib_tac = params['calib_tac']
        else:
            set_params(child, **params)
            
