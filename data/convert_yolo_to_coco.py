from pylabel import importer
import json
import os
import pandas as pd

# Define paths
path_to_annotations = r"H:\Shared drives\RnD\Data\Robot-controlled A\as_yolo\labels"
rel_path_to_images = "..\\images\\"
output_path = r"H:\Shared drives\RnD\Data\Robot-controlled A\as_yolo_as_coco_annotations.json"
intermediate_df_path = r"H:\Shared drives\RnD\Data\Robot-controlled A\as_yolo_df.feather"

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
        cat_names=[f"botA_tl#{subdir_name[-1]}"]
    )

    # TODO: this doesn't do what we want for classes; all annotations end up being of category "botA_tl#0"

    datasets.append(dataset)

# Collect all DataFrames
dfs = [ds.df for ds in datasets]

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Write concatenated DataFrame to disk
combined_df.to_feather(intermediate_df_path)  # Requires pyarrow!

# Create a new dataset object with the combined DataFrame
combined_dataset = importer.Dataset(combined_df)

# Verify the combined dataset
print(combined_dataset)
print(combined_dataset.df.head())

# Export to COCO format
combined_dataset.export.ExportToCoco(
    cat_id_index=0,
    output_path=output_path
)

# Verify the COCO format annotations
with open(output_path, 'r') as f:
    coco_annotations = json.load(f)

print(coco_annotations['annotations'][0])

