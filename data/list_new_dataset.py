# https://chatgpt.com/share/67188e15-3b80-8003-b0e6-a39dd3fb3ed2
# NOTE: PyLabel fails to correctly export to MS COCO format images with nothing but background, so we'll do without

import os

from tools.dataset_converters.images2coco import collect_image_infos, cvt_to_coco_json
import mmcv


images_path = r"H:/Shared drives/RnD/Data/Lab E/whitebalanced"
annotations_path = r"H:/Shared drives/RnD/Data/Lab E/whitebalanced_as_coco_annotations.json"

# 1 load image list info
print(f'Collecting image infos from {images_path}')
img_infos = collect_image_infos(images_path)

# exclude mask images
img_infos = [img_info for img_info in img_infos if img_info['filename'].find('mask') == -1]

for img_info in img_infos:
    img_info['filename'] = os.path.split(img_info['filename'])[1]

# sort by filename
img_infos = sorted(img_infos, key=lambda x: x['filename'])

# 2 convert to coco format data
classes = [f'AR{i:02}' for i in range(1, 16)]
print(f"Classes: {classes}")
coco_info = cvt_to_coco_json(img_infos, classes)

# 3 dump
print(f'save json file: {annotations_path}')
mmcv.dump(coco_info, annotations_path, indent=4)

