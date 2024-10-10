from pylabel import importer
import json
import os
import pandas as pd

# TODO: replace all the slashes to proper POSIX ones

# Define paths
path_to_annotations = r"H:\Shared drives\RnD\Data\Robot-controlled A\as_yolo\labels"
rel_path_to_images = "..\\images\\"
output_path = r"H:\Shared drives\RnD\Data\Robot-controlled A\as_yolo_as_coco_annotations.json"
intermediate_df_path = r"H:\Shared drives\RnD\Data\Robot-controlled A\as_yolo_df.feather"
category_names = ['class0'] + [f"AR{i:02}" for i in range(1, 16)]

# Initialize a list to hold dataset objects
datasets = []

# Get all subdirectories within your images directory
subdirs = [d for d in os.listdir(path_to_annotations) if os.path.isdir(os.path.join(path_to_annotations, d))]

# Iterate over all subdirectories
for subdir_name in subdirs:
    labels_subdir = os.path.join(path_to_annotations, subdir_name)
    print(f'Processing "{labels_subdir}"')

    # Check if the corresponding labels subdirectory exists
    images_subdir = os.path.join(path_to_annotations, rel_path_to_images, subdir_name)
    if not os.path.exists(images_subdir):
        print(f"Warning: Images subdirectory {images_subdir} does not exist.")
        continue

    # Import YOLO annotations from the subdirectory
    dataset = importer.ImportYoloV5(
        path=labels_subdir,
        path_to_images=os.path.join('..', rel_path_to_images, subdir_name),
        cat_names=category_names
    )

    class_id = int(subdir_name.split(' ')[1])
    dataset.df['cat_id'] = class_id
    dataset.df['cat_name'] = category_names[class_id]

    datasets.append(dataset)

# Collect all DataFrames
dfs = [ds.df for ds in datasets]

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)
assert len(combined_df) == sum(len(df) for df in dfs)

# Write concatenated DataFrame to disk
combined_df.to_feather(intermediate_df_path)  # Requires pyarrow!

# Modify paths to images in the combined DataFrame to be relative to the root directory
combined_df['img_folder'] = combined_df['img_folder'].apply(lambda _p: os.path.split(_p)[-1])

# Assign unique IDs to each record
combined_df['img_id'] = range(len(combined_df))

# Create a new dataset object with the combined DataFrame
combined_dataset = importer.Dataset(combined_df)

# Verify the combined dataset
print(combined_dataset)
print(combined_dataset.df.head())
print(combined_dataset.df.tail())

# Export to COCO format
combined_dataset.export.ExportToCoco(
    cat_id_index=0,
    output_path=output_path
)

# Verify the COCO format annotations
with open(output_path, 'r') as f:
    coco_annotations = json.load(f)

assert len(coco_annotations['images']) == sum(len(df) for df in dfs)
print(coco_annotations['annotations'][0])

