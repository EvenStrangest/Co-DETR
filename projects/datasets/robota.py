from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class RobotaDataset(CocoDataset):

    CLASSES = \
        ('AR01', 'AR02', 'AR03', 'AR05', 'AR06', 'AR07', 'AR09', 'AR10', 'AR12', 'AR13', 'AR14', 'AR15', 'AR16')

    CLASSES_for_visualization = (
        'AR01 Foerster Sponge',
        'AR10 Towel Forceps 13.2',
        'AR12 MIXTER Forceps  21.6',
        'AR13 Michigan MIXTER Forceps 23.5',
        'AR02 Metzenbaum-Nelson Scissors',
        'AR03 Dissecting Scissors 23.0',
        'AR05 Metzenbaum Scissors, straight 20.0',
        'AR06 POTTS-SMITH  19.1',
        'AR07 Mayo Scissors 17.3',
        'AR09 Lahey Traction Forceps 15.8',
        'AR14 Septum Forceps Hartman 20.0')

    PALETTE = [
        (211, 47, 47),    # Softer red
        (56, 142, 60),    # Softer green
        (25, 118, 210),   # Softer blue
        (251, 192, 45),   # Softer yellow
        (245, 124, 0),    # Softer orange
        (123, 31, 162),   # Softer purple
        (194, 24, 91),    # Softer pink
        (0, 172, 193),    # Softer cyan
        (216, 27, 96),    # Softer magenta
        (175, 180, 43),   # Softer lime
        (0, 121, 107),    # Softer teal
        (255, 160, 0),    # Softer gold
        (255, 112, 67),   # Softer coral
        (94, 53, 177),    # Softer violet
        (198, 40, 40),    # Softer crimson
        (67, 160, 71),    # Softer spring green
    ]

