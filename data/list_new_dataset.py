# https://chatgpt.com/share/67188e15-3b80-8003-b0e6-a39dd3fb3ed2
# NOTE: an alternative would be the use of tools/dataset_converters/images2coco.py

from pylabel import importer
import pandas as pd


images_path = r"H:\Shared drives\RnD\Data\Lab A\raw_images"
annotations_path = r"H:\Shared drives\RnD\Data\Lab A\raw_images.json"

# Import images from the folder
dataset = importer.ImportImagesOnly(images_path)

# Create a dummy category
categories = pd.DataFrame([{
    'category_id': 1,
    'category_name': 'object',
    'supercategory': 'object'
}])

# Assign the dummy category to the dataset
dataset.cat_df = categories

# Create an empty annotations DataFrame with required columns
annotations_columns = ['image_id', 'category_id', 'annotation_id', 'iscrowd', 'area', 'bbox', 'segmentation']
annotations = pd.DataFrame(columns=annotations_columns)

# Assign the empty annotations DataFrame to the dataset
dataset.anno_df = annotations

# Export the dataset to COCO JSON format
dataset.export.ExportToCoco(annotations_path)
