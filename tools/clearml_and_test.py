from test import main

import os
import clearml

# create task to run remotely
task = clearml.Task.init(project_name='Co-DETR', task_name='TestCOCO', task_type=clearml.Task.TaskTypes.inference,)
# task.execute_remotely(queue_name="default")

# get MS COCO dataset from ClearML
dataset = clearml.Dataset.get(dataset_id='eaeccf28c682478c9badb6d5c5700437')

# set environment variable for the dataset path
os.environ['MMDET_DATASETS'] = dataset.get_local_copy()

# execute the main function
main()
# TODO: consider refactoring test.py to extract the argument parsing logic into a separate function

