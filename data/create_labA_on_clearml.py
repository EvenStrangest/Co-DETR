import os

from clearml import Dataset


local_root_path = r"H:\Shared drives\RnD\Data\Lab A"

# Create a dataset with ClearML`s Dataset class
dataset = Dataset.create(
    dataset_project="SurgicalTools", dataset_name="LabA", dataset_version="1.0.0"
)

dataset.add_files(path=os.path.join(local_root_path, r"raw_images"), local_base_folder=local_root_path,
                  verbose=False)
dataset.add_files(path=os.path.join(local_root_path, "raw_images.json"), local_base_folder=local_root_path,
                  verbose=True)

# Upload dataset to ClearML server
dataset.upload()

# commit dataset changes
dataset.finalize()

