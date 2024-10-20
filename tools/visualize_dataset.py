# https://chatgpt.com/share/6711f2e1-77f0-8003-9716-b0e2d6ac5a71


import os

from mmcv import Config
from mmdet.datasets import build_dataset
import projects.datasets.robota
import mmcv
import clearml
from mmdet.utils.misc import update_data_root


# Get the dataset
print("Using RobotA dataset")
# robota = clearml.Dataset.get(dataset_id='4de72c7d8fc9489fb3b1bc292b0fb0e7')
robota = clearml.Dataset.get(dataset_project='SurgicalTools', dataset_name='RobotA', dataset_version='1.2.0')
os.environ['MMDET_DATASETS'] = robota.get_local_copy() + '/'

# Load your existing configuration file
cfg = Config.fromfile('projects/configs/_base_/datasets/robota_detection.py')
update_data_root(cfg)

# Build the dataset using the configuration
dataset = build_dataset(cfg.data.train)

# Create an output directory for the images
output_dir = '/logs/visualize_dataset_robota'  # You can change this to your preferred directory
os.makedirs(output_dir, exist_ok=True)

# Iterate over the dataset and visualize images with bounding boxes
for idx in range(len(dataset)):
    data = dataset[idx]

    # Get the image tensor and convert it to a NumPy array
    img = data['img'].data.permute(1, 2, 0).numpy()

    # Get bounding boxes and labels
    bboxes = data['gt_bboxes'].data.numpy()
    labels = data['gt_labels'].data.numpy()

    # Copy the image to avoid modifying the original
    img_show = img.copy()

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
