# Top-Theta Attention
Implementation and evaluation **Top-Theta Attention** and **Top-k Attention** on Large Language Models, as presented in the article [Top-Theta Attention: Sparsifying Transformers by Compensated Thresholding](TODO)

Testing was done for the LLaMA models on `arc_challenge/hellaswag/arc_easy` datasets for Q&A evaluation (prefill-only tasks) and on humaneval dataset (prefill + generative decoding). For detailed tested variants, refer to [reproduce.md](reproduce.md).

## Install

```bash
# Create a conda/pyenv virtual environment in the local directory
conda create python=3.9.12 --prefix ./topksandbox
conda activate $(pwd)/topksandbox

# Install the human_eval repo and enable unsandboxed evaluation of LLM-generated python programs
git clone https://github.com/openai/human-eval.git
pushd human-eval
sed -i 's/^#\s*\(.*exec(check_program, exec_globals).*\)/\                        exec(check_program, exec_globals)/'  human_eval/execution.py
sed -i '44a \    num_tasks:int = 164,' human_eval/evaluation.py
sed -i 's/^.*assert len(completion_id) == len(problems), "Some problems are not attempted."/\        # assert len(completion_id) == len(problems), "Some problems are not attempted."/' human_eval/evaluation.py
pip install -e .
popd

# Install the human_eval repo and patch it with the calibration tasks for Hellaswag, Arc_Challenge, and Arch_Easy
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
pushd lm-evaluation-harness
git checkout e5dfd0304efa6668bcf61aa3eb22ce8abe337eaf
echo -e "include: hellaswag.yaml\ntask: hellaswag_calibration\ntest_split: train\n" > lm_eval/tasks/hellaswag/hellaswag_calibration.yaml
echo -e "include: arc_challenge.yaml\ntask: arc_challenge_calibration\ntest_split: train\n" > lm_eval/tasks/arc/arc_challenge_calibration.yaml
echo -e "include: arc_easy.yaml\ntask: arc_easy_calibration\ntest_split: train\n" > lm_eval/tasks/arc/arc_easy_calibration.yaml
pip install -e .
popd

# Install the current topk_attention repo with its dependencies
# git clone <HERE COMES THE GITHUB REPO>topk_attention.git or just have the topk_attention directory ready with the code
pushd topk_attention
pip install -r requirements.txt
popd

```

## Entry point

1. `test_llama.py` - Runs Q&A task (hellaswag, arc, mmlu) evaluations for Top-k and Top-threshold (and baseline) on Llama models.
2. `gen_llama.py` - Runs text generation task (humaneval) evaluations for Top-k and Top-threshold (and baseline) on Llama models.

## Implementation Details

`topk_llama.py` contains the implementation of `TopK_LLamaAttention` class, which can replace the `LlamaAttention` layer from the transformers library. The `TopK_LLamaAttention` class implements all the functionality of Top-k attention and of Top-threshold attention (including calibration) and numerical compensations. A few usage details:
 
1. `mode=0` implements only Top-threshold, `mode=1` implements only Top-k, any other mode implements the baseline.

2. `update_model(model)` - function to replace all `LlamaAttention` with `TopK_LLamaAttention` layers.

3. `set_params(model, **params)` - function to set the parameters to the `TopK_LLamaAttention` layer e.g. mode, K etc.

## Evaluations

`test_llama.py` and `gen_llama.py` - runs evaluations for Top-k and Top-threshold on Llama models (supported models: llama2-7b, llama2-70b, codellama-34b, llama-3-8B, llama-3-8B-Instruct, llama-3-70B, llama-3-70B-Instruct).


Outputs:
* Results are logged in the _results-*_ directories. 
* Various products of the run are dumped to a dedicated and time-stamped sub-directory under the _products_ directory.

### Example of running an evaluation

1. Evaluate llama2-7b on hellaswag task using top-threshold attention (`--mode 0`) calibrated for k=64 (`--k 64`) for all layers except layer 0 and 1 (`--layerk 0:512,1:512`), where the threholding should be placed before the softmax (`--placement pre-softmax`). During the calibration, for every (layer,head,seqlen) determine an individual threshold value by taking the average threshold across the thresholds found at the different calibration samples and increase this average by 0.1 standard deviation (`--calib_add_sigma 0.1`). In addition, during the calibration apply the recommended topk-at-calibration feature (`--calib_tac`) to emulate the presence of thresholding. During the inference, apply softmax denominator compensation of the type "offline-calibrated" (`--sdc offline-calibrated`) and use V-mean compensation (`--vmc`). The option `--timestamps` could become default in the future, but for now it is required to specify it in order to create a separate products subdirectory for the files being dumped during the evaluation run. 
```bash
python test_llama.py --timestamps --llama 2-7 --task hellaswag --mode 0 --k 64 --layerk 0:512,1:512 --placement pre-softmax --calib_tac --calib_add_sigma 0.1 --sdc offline-calibrated --vmc
```

2. Evaluate codellama-34b model with top-threshold attention (`--mode 0`) calibrated for k=64 (`--k 64`) for all layers except layer 0 and 1 (`--layerk 0:512,1:512`), where the threholding should be placed before the softmax (`--placement pre-softmax`). The test set consists of the first 20 out of 167 test examples (tasks) of the humaneval dataset. Evaluate only a single output per task (quality metric will be pass@1). No SDC or VMC compensations are applied. The model is allowed to generate tokens until the EOS token is generated or until the total sequence length reaches 2048 before being halted.
```bash
python gen_llama.py --timestamps --llama 34 --mode 0 --k 64  --layerk 0:512,1:512 --placement pre-softmax --calib_add_sigma 0.1 --calib_sample_frac 1.0 --calib_tac  --num_samples_per_task 1 --max_seq_len 2048
```

## Plotting

`plot_th_llama.py` - Plots the calibrated thresholds for different layers & Attention matrix size required w.r.t Top-threshold during evaluation using calibrated thresholds. The produced plots are written to the products subdirectory of the evaluation (`-d`) and all have the title specified after the argument `-t`.

```bash
python plot_th_llama.py -d "products/2024-04-29_17-29-01_774054" -t "CodeLLaMA-34b-arc_challenge Top-th pre-softmax (k=512,512,128,128,...) single-k calibration mean+1.0*sigma, nacc=52.39% (base=54.44%)"`
```

`plot_gen_llama.py` - important plots of the prompt and completion lengths

**Important results visualizations:** Check out the [notebooks](notebooks/) for various visualization capabilities.

## References

1. https://github.com/EleutherAI/lm-evaluation-harness 
2. https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard 
3. https://github.com/openai/human-eval
