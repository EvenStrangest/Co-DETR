import os

from pylabel import importer
from clearml import Dataset


local_root_path = r"H:\Shared drives\RnD\Data\Robot-controlled A"


# Import COCO-format annotations
annotations = importer.ImportCoco(path=os.path.join(local_root_path, "as_yolo_as_coco_annotations.json"),
                                  path_to_images='as_yolo/images')

# Split the dataset into training and validation sets
annotations.split(0.8)

# Create a dataset with ClearML`s Dataset class
dataset = Dataset.create(
    dataset_project="SurgicalTools", dataset_name="RobotA",
    dataset_version="1.1.0",
    parent_datasets=["018f56fde567441b987451c695f0f629"]
)

dataset.add_files(path=os.path.join(local_root_path, "as_yolo_as_coco_ann_train.json"), local_base_folder=local_root_path,
                  verbose=True)
dataset.add_files(path=os.path.join(local_root_path, "as_yolo_as_coco_ann_val.json"), local_base_folder=local_root_path,
                  verbose=True)

assert len(dataset.list_modified_files()) == 0
assert len(dataset.list_added_files()) == 2
assert len(dataset.list_files()) == 50528 + 1 + 2

# Upload dataset to ClearML server
dataset.upload()

# commit dataset changes
dataset.finalize()

