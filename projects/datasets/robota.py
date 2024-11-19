from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class RobotaDataset(CocoDataset):

    CLASSES = ('AR01', 'AR10', 'AR12', 'AR13', 'AR02', 'AR03', 'AR05', 'AR06', 'AR07', 'AR09', 'AR14')

