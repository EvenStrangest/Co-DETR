import os
import sys

from clearml import Task
import clearml

from train import main, parse_args


if os.environ.get('CHOICE_DATASET') == 'RobotA':
    # set the project name
    task_name = 'FinetuneRobotA'
else:
    # set the project name
    task_name = 'FinetuneCOCO'

# create task to run remotely
task = Task.init(project_name='Co-DETR', task_name=task_name, task_type=clearml.Task.TaskTypes.training,
                 deferred_init=False, )
task.set_base_docker(docker_image='361432929675.dkr.ecr.us-east-1.amazonaws.com/trackimed/co_detr_manual:2024OCT06',
                     docker_arguments=f'--env CHOICE_DATASET={os.environ.get("CHOICE_DATASET")}',
                     docker_setup_bash_script='')

args = parse_args()

# task.execute_remotely(queue_name="default")
print(f"sys.argv: {sys.argv}")

if os.name == 'posix':
    # list the mounted file systems
    os.system('df -h')
    # list files in /checkpoints directory
    os.system('ls -l /checkpoints')

# set environment variable for the dataset path
if os.environ.get('CHOICE_DATASET') == 'RobotA':
    print("Using RobotA dataset")
    # robota = clearml.Dataset.get(dataset_id='4de72c7d8fc9489fb3b1bc292b0fb0e7')
    robota = clearml.Dataset.get(dataset_project='SurgicalTools', dataset_name='RobotA', dataset_version='1.2.0')
    os.environ['MMDET_DATASETS'] = robota.get_local_copy() + '/'
else:
    print("Using MS COCO dataset")
    # mscoco = clearml.Dataset.get(dataset_id='eaeccf28c682478c9badb6d5c5700437')
    mscoco = clearml.Dataset.get(dataset_project='MS_COCO', dataset_name='MS_COCO_2017', dataset_version='1.0.0')
    os.environ['MMDET_DATASETS'] = mscoco.get_local_copy() + '/'

# execute the main function
main()
# TODO: consider refactoring train.py to extract the argument parsing logic into a separate function

