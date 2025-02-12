"""
Utility functions and configurations for lm-eval tasks
"""
from typing import Tuple

def get_task_config(task: str) -> Tuple[int, int, int]: 
    """
    Gets the name of the lm-eval harness's task.
    Return the following configuration parameters:
    - num_fewshot: setting to be passed to lm-eval
    - calibration_samples: number of dataset's examples (e.g. questions, in Q&A
      tasks) to perform the calibration on. 
    - calibration_requests: number of actual inference runs to be performed
      For example: when there are 4 possible answers per question, that the
      model needs to choose from, there will be num_samples * 4 inference 
      requests made.
    """
    if task.startswith('arc_'):
        num_fewshot = 25
        calibration_samples = 120  # ~10% of the test dataset
        if task=='arc_challenge':
            calibration_requests = calibration_samples * 4
        elif task=='arc_easy':
            calibration_requests = calibration_samples * 4 - 1  # peculiarity of this task is to have one less request
        else:
            raise NotImplementedError("only arc_challenge and arc_easy tasks are supported from the ai2_arc challenge tasks.")

    elif task == 'hellaswag':
        num_fewshot = 10
        calibration_samples = 0.1
        calibration_requests = 4016

    elif 'mmlu' == task:
        num_fewshot=5
        calibration_samples = 0.1  # 10% of *each* of the 57 tasks in the dataset
        calibration_requests = 3648  # the sum of all inferences in all tasks (that will be made for the first 10% of the examples)

    elif 'gsm8k' == task:
        num_fewshot=8
        calibration_samples = 120 # TODO: change to ??? to become 10% of the entire dataset
        calibration_requests = calibration_samples * 4  # TODO: change to a correct value, based on the number of inference runs made for the calibration_samples   
    
    return num_fewshot, calibration_samples, calibration_requests