from mmcv.utils import print_log
import mmcv


def fetch_clearml_datasets(cfg, logger=None):
    """Fetch datasets from ClearML and update the paths"""

    assert isinstance(cfg, mmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'

    if 'MMDET_DATASETS' in os.environ:
        dst_root = os.environ['MMDET_DATASETS']
        print_log(f'MMDET_DATASETS has been set to be {dst_root}.'
                  f'Using {dst_root} as data root.')
    else:
        return

    def update(cfg, src_str, dst_str):
        for k, v in cfg.items():
            if isinstance(v, mmcv.ConfigDict):
                update(cfg[k], src_str, dst_str)
            if isinstance(v, list):
                for i, _ in enumerate(v):
                    if isinstance(v[i], mmcv.ConfigDict):
                        update(v[i], src_str, dst_str)
                    elif isinstance(v[i], str) and src_str in v[i]:
                        v[i] = v[i].replace(src_str, dst_str)
            if isinstance(v, str) and src_str in v:
                cfg[k] = v.replace(src_str, dst_str)

    update(cfg.data, cfg.data_root, dst_root)
    cfg.data_root = dst_root
