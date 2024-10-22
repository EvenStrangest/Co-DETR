from pylabel import importer

from data.split_robotA_on_clearml import annotations

images_path = r"H:\Shared drives\RnD\Data\Lab A\raw_images"
annotations_path = r"H:\Shared drives\RnD\Data\Lab A\raw_images.json"

# Import images from the folder
dataset = importer.ImportImages(images_path)

# Export the dataset to COCO JSON format
dataset.export.ExportToCoco(annotations_path)
