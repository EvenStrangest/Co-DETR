from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class RobotaDataset(CocoDataset):

    CLASSES = ('AR01 Foerster Sponge',
               'AR10',
               'AR12',
               'AR13',
               'AR02 Metzenbaum-Nelson Scissors',
               'AR03 Dissecting Scissors 23.0',
               'AR05 Metzenbaum Scissors, straight 20.0',
               'AR06 POTTS-SMITH  19.1',
               'AR07 Mayo Scissors 17.3',
               'AR09 Lahey Traction Forceps 15.8',
               'AR14 Septum Forceps Hartman 20.0')

