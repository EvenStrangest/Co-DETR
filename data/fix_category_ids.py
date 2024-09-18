import json

inplace_edit_json_path = r"H:\Shared drives\RnD\Data\Robot-controlled A\as_yolo_as_coco_annotations.json"

with open(inplace_edit_json_path, 'r') as f:
    annotations = json.load(f)

# Set the category id to be the (numeric value) of the digits before the underscore in the corresponding image filename
for ann in annotations['annotations']:
    img_id = int(ann['image_id'])
    img_record = annotations['images'][img_id]
    assert int(img_record['id']) == img_id
    img_filename = img_record['file_name']
    cat_id = int(img_filename.split('_')[0])
    ann['category_id'] = cat_id

with open(inplace_edit_json_path, 'w') as f:
    json.dump(annotations, f, indent=4)

