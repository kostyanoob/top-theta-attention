#%%
# First of all create a unique time stamp for the current run
from datetime import datetime
timestamp = str(datetime.now()).replace(' ','_').replace('.','_').replace(':','-')
import os
import logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename=f"logs/{timestamp}-app.log", filemode='w', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import argparse
import lm_eval.models
import lm_eval.evaluator
import glob
import task_util
from util import *
from topk_llama import *

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--llama',
    type=str,
    default='2-7',
    choices=['2-7','3-8','3-8i','34','2-70','3-70','3-70i'],
    help='llama model alias to be loaded. specifying "i" after the number implies "Instruct-finetuned"')
parser.add_argument(
    '--mode',
    type=str,
    default='3',
    choices=['0','1','3'],
    help='0 - TH, 1 - TOPK, other number - BASELINE')
parser.add_argument(
    '--k',
    metavar='K',
    type=str,
    default='256',
    help='k parameter, used only for --mode 0 or --mode 1')
parser.add_argument(
    '--calib_load_path',
    type=str,
    default="",
    help='path to a directory containing th.txt or sdc.txt (applicable '
         'only for mode=0 or when --sdc=offline-calibrated is used) If '
         'this argument is specified, then calibration will not take '
         'place, but rather loaded from the existing files.')
parser.add_argument(
    "--calib_tac", 
    help="Top-K At Calibrating. This would incorporate the "
         "sparsification effect already during the calibration",
    action="store_true")
parser.add_argument(
    '--calib_add_sigma',
    type=float,
    default=0.0,
    help='this amount of standard deviations will be added to the '
         'average threshold when at the end of the calibration the '
         'aggregating of the threshold obained from the various '
         'calibraion samples takes place. Has effect only for mode=0.')
parser.add_argument(
    '--calib_sample_frac',
    type=float,
    default=0.1,
    help='percentage of queries (i.e. attention rows) to use for calibration')
parser.add_argument(
    '--layerk',
    metavar='DICT',
    type=str,
    default='',
    help='dictionary that overwrites the --k for chosen layers. E.g 3:128 '
         'forces the 3rd layer to use k=128')
parser.add_argument(
    '--task',
    type=str,
    default='hellaswag',
    choices=['hellaswag', 'arc_challenge', 'arc_easy', 'mmlu', 'gsm8k'],
    help='evaluation task in the lm eval harness.')
parser.add_argument(
    '--num_samples',
    type=float,
    default=None,
    help='number of samples to use for the evaluation (if <1 then treated as a '
         'fraction of all samples). Default is using all samples.')
parser.add_argument(
    "--timestamps", 
    help="append timestamps to the filename of the cache-database (deleted at "
         "the end of the run) and to the products directory name. Always use "
         "this option when running multiple evaluations in parallel, in order "
         "to avoid database collisions.",
    action="store_true")
parser.add_argument(
    '--placement',
    type=str,
    default='pre-softmax',
    choices=['pre-softmax', 'post-softmax', 'none'],
    help='where to apply the top-k/threshold. none - stands for "do not apply"')
parser.add_argument(
    '--sdc',
    type=str,
    default='none',
    choices=['none', 'exact', 'exp-threshold', 'offline-calibrated'],
    help='software denominator compensation: method of compensating the softmax '
         'denominator, which becomes incomplete as a result of top-k/thresholding '
         'applied before the softmax. none - do nothing, exact - compute precisely '
         'the missing denominator term, exp-threshold applies only for the '
         'thresholding method, the two calibrated methods apply both for topk '
         'and thresholding.')
parser.add_argument(
    '--sdc_scale',
    type=float,
    default=1.0,
    help='multiplicative factor applied to the missing term, which is added by '
         'the compensation set by --sdc option. Has no effect when --sdc is none')    
parser.add_argument(
    "--vmc", 
    help="v-mean compensation. Adds a mean-row of V matrix (average across V's columns) "
         "to every output vector of the self_attention. The addition is weighted by "
         "(1-softmax_row.sum()). This option only has effect when the attention rows do not "
         "sum up to 1. Therefore, it is worth applying vmc only with --mode 0 or 1. Note that "
         "when --placement is set to 'pre-softmax' then it is necessary to apply --sdc "
         "different than 'none' in order to have the softmax output summed up to less than 1.",
    action="store_true")
parser.add_argument(
    "--capk", 
    help="cap the number of elements that pass the trhesholding to at most k in "
         " every attention row, prioritizing the last (most recent) tokens. "
         "Applicable only in mode=0.",
    action="store_true")

# Check validity of cmd-line arguments
args = parser.parse_args()
layer_k_dict = dict(eval('{'+args.layerk+'}'))
placement = args.placement if args.mode in {'0','1'} else 'none'

# Avoid further usage of the timestamp for the threshold products and for the lm-eval cache database 
# Note that the logging will anyway be using time stamp
if not args.timestamps:
    timestamp = ""

# Note to metrics, check out --> https://blog.eleuther.ai/multiple-choice-normalization/
num_attn_layers = get_model_num_attn_layers(args.llama)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model_name, model_shortname = get_model_names(args.llama)
cache_prefix_path = f"lm_cache/{model_name}{timestamp}"
products_dir_path = f"products/{timestamp}"
os.makedirs(products_dir_path, exist_ok=True)
os.makedirs("lm_cache", exist_ok=True)
os.makedirs("results-Llama", exist_ok=True)
device="cuda"
device_map_option="auto"
dtype="float16"
batch_size=1
model_type = 'causal' # seq2seq for enc-dec models
        # Based on https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard :: About

num_fewshot, calibration_samples, calibration_requests = task_util.get_task_config(args.task)
calibration_samples = 0.025159755 if args.task == 'hellaswag' and num_attn_layers>32 else calibration_samples # less calibration for the large models (to expedite the calibration)

tasks_list=[args.task]
tasks_calibration_list = [t_ + "_calibration" for t_ in tasks_list]  # calibration suffix points to a task with a data split different from "test"

# Check validity of cmd-line arguments
assert(args.mode not in ['0','1'] or args.placement == 'pre-softmax' or args.sdc == 'none') # softmax compensation should be only when top-k/th is applied and only if it's applied pre-softmax
assert(args.mode == '0' or args.sdc != 'exp-threshold') # softmax compensation of exponentiated threshold can be only applied jointly with top-threshold pre-softmax
assert(not args.vmc or args.mode in ['0','1']) # v-mean compesation is applicatble only when sparsification is applied (mode 0 or 1)
assert(not (args.vmc and args.mode in ['0','1'] and args.placement == 'pre-softmax') or args.sdc != 'none') # v-mean compesation in a pre-softmax sparsifiation placement requires having a software denominator compensation to remain meaningful (without sdc: 1-softmax_row.sum() == 0)
assert(not(args.calib_tac) or args.mode == '0')  # topk-at-calibration should be applied only on mode=0 (thresholding)
assert(not(args.calib_load_path!="") or args.mode == '0' or args.sdc == 'offline-calibrated')
assert(not(args.capk) or args.mode == '0')  # --capk can be used only with top-threshld (mode=0)
assert(0.0 <= args.calib_sample_frac <= 1.0)

print(f"RUN TIMESTAMP:{timestamp}")

# Load the model from checkpoint
lm_eval.tasks.initialize_tasks()
model = get_model(model_type, model_name, device, dtype, batch_size, device_map_option)

# Update the vanilla model's LlamaDecoderLayer layers with top-k parameters
update_model(model.model, False, products_dir_path) # this one changes dataformat to float32 again

def run_model(model, tasks_list, num_fewshot, batch_size, n_samples=None):
    eval_dict = lm_eval.evaluator.simple_evaluate(
        model=model,
        model_args="",
        tasks=tasks_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device='cuda',
        use_cache=cache_prefix_path,
        limit=n_samples,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=True,
    )
    return eval_dict

# Evaluate LLama
# mode-0 (TH) mode-1 (TOPK) mode-anything other than 0/1 (BASELINE)
for mode in eval('list({'+args.mode+'})'):#0, 1, 3]: 
    K_list = eval('list({'+args.k+'})')
    if not (mode in [0,1]):
        K_list = [999,]
    for K in K_list:
        K_list=list([K]*num_attn_layers) # List with num_attn_layers elements, each value K
        for key, value in layer_k_dict.items():
            K_list[key]=value
        print(K_list)
        
        if args.calib_load_path != "":
            # load the parameters and the pre-reviously calibrated thresholds from a specified directory
            set_params(model.model, K=K_list, mode=mode, placement=placement,
                       sdc=args.sdc,
                       sdc_scale=args.sdc_scale,
                       vmc=args.vmc,
                       calib_load_path=args.calib_load_path,
                       calibrate=False,
                       capk=args.capk,
                       test_layer=None)
            
        elif mode==0 or args.sdc == "offline-calibrated":
            # calibration is needed prior to an evaluation  
            set_params(model.model, K=K_list, mode=mode, placement=placement,
                       sdc=args.sdc,
                       sdc_scale=args.sdc_scale,
                       vmc=args.vmc,
                       calib_load_path="",
                       calibrate=True, 
                       calib_tac=args.calib_tac,
                       calib_add_sigma=args.calib_add_sigma,                       
                       calib_sample_frac=args.calib_sample_frac, 
                       calibration_requests=calibration_requests, 
                       capk=args.capk,
                       test_layer=None) 
            eval_dict = run_model(model, tasks_calibration_list, num_fewshot, batch_size, n_samples=calibration_samples)
            os.remove(f"{cache_prefix_path}_rank0.db")
            
        # Run
        # for test_layer in [None]: <---- Use this to test for all layers - i.e. Top-K/TH for all layers
        for test_layer in [None]: #for test_layer in range(num_attn_layers):
            set_params(model.model, K=K_list, mode=mode,  placement=placement,
                       sdc=args.sdc,
                       sdc_scale=args.sdc_scale,
                       vmc=args.vmc,
                       calib_load_path="",
                       calibrate=False,
                       capk=args.capk,
                       test_layer=test_layer)
            print(model.model)
            eval_dict = run_model(model, tasks_list, num_fewshot, batch_size, n_samples=args.num_samples)
            table = lm_eval.evaluator.make_table(eval_dict)
            with open(f'results-Llama/{model_shortname}_{args.task}_mode{mode}_placement{placement}.txt', 'a') as f:
                f.write(timestamp + '\n')
                f.write(f'K:{K_list} mode:{mode} layer:{test_layer} placement:{placement} sdc:{args.sdc} sdc_scale:{args.sdc_scale} vmc:{args.vmc} capk:{args.capk} calib_tac:{args.calib_tac} calib_add_sigma:{args.calib_add_sigma} calib_sample_frac:{args.calib_sample_frac} calib_load_path:{args.calib_load_path}\n')
                f.write(table + '\n')
                os.remove(f"{cache_prefix_path}_rank0.db")

        # Compress the products if any were produced
        layer_csv_txt_files_paths = glob.glob(os.path.join(products_dir_path, 'layer*.[ct][sx][vt]'))
        compress_files_parallel(layer_csv_txt_files_paths, remove_uncompressed=True, num_cores=164)

