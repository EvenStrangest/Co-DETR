from train import main

import os
import clearml


# create task to run remotely
task = clearml.Task.init(project_name='Co-DETR', task_name='FinetuneCOCO', task_type=clearml.Task.TaskTypes.inference,)
# task.execute_remotely(queue_name="default")

# get MS COCO dataset from ClearML
mscoco = clearml.Dataset.get(dataset_id='eaeccf28c682478c9badb6d5c5700437')
robota = clearml.Dataset.get(dataset_id='018f56fde567441b987451c695f0f629')

# set environment variable for the dataset path
if os.environ.get('CHOICE_DATASET') == 'RobotA':
    os.environ['MMDET_DATASETS'] = robota.get_local_copy() + '/'
else:
    os.environ['MMDET_DATASETS'] = mscoco.get_local_copy() + '/'

# execute the main function
main()
# TODO: consider refactoring train.py to extract the argument parsing logic into a separate function

