import os

from clearml import Dataset


local_root_path = r"H:\Shared drives\RnD\Data\Robot-controlled A"

robota = Dataset.get(dataset_project='SurgicalTools', dataset_name='RobotA', dataset_version='1.2.0')
dataset = Dataset.create(
    dataset_project="SurgicalTools", dataset_name="RobotA_with_Photomontage_Round1", dataset_version="1.1.0",
    parent_datasets=[robota]
)

dataset.add_files(path=os.path.join(local_root_path, r"photomontages/round_1"), local_base_folder=local_root_path,
                  verbose=False)
dataset.add_files(path=os.path.join(local_root_path, "photomontages_round_1_as_coco_annotations.json"), local_base_folder=local_root_path,
                  verbose=True)

# Upload dataset to ClearML server
dataset.upload()

# commit dataset changes
dataset.finalize()

