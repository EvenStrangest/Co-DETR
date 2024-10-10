from train import main

import os
import clearml


# create task to run remotely
task = clearml.Task.init(project_name='Co-DETR', task_name='FinetuneCOCO', task_type=clearml.Task.TaskTypes.inference,
                         deferred_init=False, )
task.set_base_docker(docker_image='361432929675.dkr.ecr.us-east-1.amazonaws.com/trackimed/co_detr_manual:2024OCT06',
                     docker_arguments='',
                     docker_setup_bash_script='')
# task.execute_remotely(queue_name="default")

# list the mounted file systems
os.system('df -h')
# list files in /checkpoints directory
os.system('ls -l /checkpoints')

# get MS COCO dataset from ClearML
mscoco = clearml.Dataset.get(dataset_id='eaeccf28c682478c9badb6d5c5700437')
robota = clearml.Dataset.get(dataset_id='4de72c7d8fc9489fb3b1bc292b0fb0e7')

# set environment variable for the dataset path
if os.environ.get('CHOICE_DATASET') == 'RobotA':
    print("Using RobotA dataset")
    os.environ['MMDET_DATASETS'] = robota.get_local_copy() + '/'
else:
    print("Using MS COCO dataset")
    os.environ['MMDET_DATASETS'] = mscoco.get_local_copy() + '/'

# execute the main function
main()
# TODO: consider refactoring train.py to extract the argument parsing logic into a separate function

