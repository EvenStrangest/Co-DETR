# NOTE: an alternative would be the use of tools/dataset_converters/images2coco.py

from pylabel import importer


images_path = r"H:\Shared drives\RnD\Data\Lab A\raw_images"
annotations_path = r"H:\Shared drives\RnD\Data\Lab A\raw_images.json"

# Import images from the folder
dataset = importer.ImportImages(images_path)

# Export the dataset to COCO JSON format
dataset.export.ExportToCoco(annotations_path)
