# Notebooks 

These notebooks are used to summarize and visualize various results and statistics gathered after running test_llama.py or gen_llama.py.

1. `notebooks/1-size.ipynb` - provides a concise comparison of effective attention size for a set of evaluations
2. `notebooks/2-th_histogram.ipynb` - plots the distribution of the thresholds observed during the calibration, and superimposes the selected final threshold.
3. `notebooks/3-vrow_populatiry.ipynb` - plots the popularity of each token (as pointed by top-k or top-theta) in a group of  heads (if GQA is used for the attenntion layer)
4. `notebooks/4-accuracy-kept_attn-kept_vrow-tradeoff_QA.ipynb` - showing tradeoff between accuracy and number of selected attention tokens - for QA tasks (heallsawag, arc, ...)
5. `notebooks/5-accuracy-kept_attn-kept_vrow-tradeoff_humaneval.ipynb` - showing tradeoff between accuracy and number of V-rows to keep - for code generation task (human_eval)