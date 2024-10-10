import os

from clearml import Dataset


coco_path = "C:/Users/asafe/PycharmProjects/data/coco/"

# Create a dataset with ClearML`s Dataset class
dataset = Dataset.create(
    dataset_project="MS_COCO", dataset_name="MS_COCO_2017"
)

dataset.add_files(path=os.path.join(coco_path, "annotations/instances_val2017.json"), local_base_folder=coco_path,
                  verbose=True)
dataset.add_files(path=os.path.join(coco_path, "val2017/"), local_base_folder=coco_path,
                  verbose=False)

# Upload dataset to ClearML server
dataset.upload()

# commit dataset changes
dataset.finalize()

