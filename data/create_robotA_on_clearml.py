import os

from clearml import Dataset


local_root_path = r"H:\Shared drives\RnD\Data\Robot-controlled A"

# Create a dataset with ClearML`s Dataset class
dataset = Dataset.create(
    dataset_project="SurgicalTools", dataset_name="RobotA"
)

dataset.add_files(path=os.path.join(local_root_path, r"as_yolo/images"), local_base_folder=local_root_path,
                  verbose=False)
dataset.add_files(path=os.path.join(local_root_path, "as_yolo_as_coco_annotations.json"), local_base_folder=local_root_path,
                  verbose=True)

# Upload dataset to ClearML server
dataset.upload()

# commit dataset changes
dataset.finalize()

