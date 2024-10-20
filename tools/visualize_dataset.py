# https://chatgpt.com/share/6711f2e1-77f0-8003-9716-b0e2d6ac5a71


import os

from mmcv import Config
from mmdet.datasets import build_dataset
import projects.datasets.robota
import mmcv
import clearml
from mmdet.utils.misc import update_data_root
from tqdm import tqdm


# Get the dataset
if os.environ.get('CHOICE_DATASET') == 'RobotA':
    print("Using RobotA dataset")
    # robota = clearml.Dataset.get(dataset_id='4de72c7d8fc9489fb3b1bc292b0fb0e7')
    robota = clearml.Dataset.get(dataset_project='SurgicalTools', dataset_name='RobotA', dataset_version='1.2.0')
    os.environ['MMDET_DATASETS'] = robota.get_local_copy() + '/'

    cfg = Config.fromfile('projects/configs/_base_/datasets/robota_detection.py')

    output_dir = '/logs/visualize_dataset_robota'
else:
    print("Using MS COCO dataset")
    # mscoco = clearml.Dataset.get(dataset_id='eaeccf28c682478c9badb6d5c5700437')
    mscoco = clearml.Dataset.get(dataset_project='MS_COCO', dataset_name='MS_COCO_2017', dataset_version='1.0.0')
    os.environ['MMDET_DATASETS'] = mscoco.get_local_copy() + '/'

    cfg = Config.fromfile('projects/configs/_base_/datasets/coco_detection.py')

    output_dir = '/logs/visualize_dataset_coco'

# Load your existing configuration file
update_data_root(cfg)

# Build the dataset using the configuration
dataset = build_dataset(cfg.data.val)

# Create an output directory for the images

os.makedirs(output_dir, exist_ok=True)

# Iterate over the dataset and visualize images with bounding boxes
for idx in tqdm(range(len(dataset))):
    data = dataset[idx]

    # Get the image tensor and convert it to a NumPy array
    img = data['img'].data.permute(1, 2, 0).numpy()
    # TODO: fails for MS COCO because it is somehow a list of tensors, whereas it is simply a tensor for RobotA

    # Get bounding boxes and labels
    bboxes = data['gt_bboxes'].data.numpy()
    labels = data['gt_labels'].data.numpy()

    # Copy the image to avoid modifying the original
    img_show = img.copy()  # TODO: getting it normalized somehow from the dataset, but imshow_det_bboxes() expects it normalized differently

    # Visualize the image with bounding boxes
    mmcv.imshow_det_bboxes(
        img_show,
        bboxes,
        labels,
        class_names=dataset.CLASSES,
        score_thr=0,
        bbox_color='green',
        text_color='green',
        thickness=2,
        font_scale=0.5,
        show=False,
        wait_time=0,
        out_file=os.path.join(output_dir, f'image_{idx}.png')  # TODO: change filename to match dataset
    )

    # Break after visualizing the first 10 images
    if idx == 9:
        break
