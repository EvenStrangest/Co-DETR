import os

from pylabel import importer
from clearml import Dataset


local_root_path = r"H:\Shared drives\RnD\Data\Robot-controlled A"


# Import COCO-format annotations
annotations = importer.ImportCoco(path=os.path.join(local_root_path, "as_yolo_as_coco_annotations.json"),
                                  path_to_images='as_yolo/images')

# Split the dataset into training and validation sets
annotations.splitter.GroupShuffleSplit(train_pct=0.8, test_pct=0.05, val_pct=0.15, group_col='img_id', random_state=1)

# Report some stuff
print(f"{annotations.analyze.class_counts=}")

# Create three instances of Dataset objects, one for each split
train_dataset = importer.Dataset(annotations.df[annotations.df['split'] == 'train'])
val_dataset = importer.Dataset(annotations.df[annotations.df['split'] == 'val'])
test_dataset = importer.Dataset(annotations.df[annotations.df['split'] == 'test'])
assert len(train_dataset.df) + len(val_dataset.df) + len(test_dataset.df) == len(annotations.df)

# Reset indices (because pylabel is iterating with an int, yach!)
train_dataset.df.reset_index(drop=True, inplace=True)
val_dataset.df.reset_index(drop=True, inplace=True)
test_dataset.df.reset_index(drop=True, inplace=True)

# Export the training, validation, and test sets to COCO format
train_dataset.export.ExportToCoco(output_path=os.path.join(local_root_path, "as_yolo_as_coco_ann_train.json"), cat_id_index=0)
val_dataset.export.ExportToCoco(output_path=os.path.join(local_root_path, "as_yolo_as_coco_ann_val.json"), cat_id_index=0)
test_dataset.export.ExportToCoco(output_path=os.path.join(local_root_path, "as_yolo_as_coco_ann_test.json"), cat_id_index=0)

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
dataset.add_files(path=os.path.join(local_root_path, "as_yolo_as_coco_ann_test.json"), local_base_folder=local_root_path,
                  verbose=True)

assert len(dataset.list_modified_files()) == 0
assert len(dataset.list_added_files()) == 3
assert len(dataset.list_files()) == 50528 + 1 + 3

# Upload dataset to ClearML server
dataset.upload()

# commit dataset changes
dataset.finalize()

