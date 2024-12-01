import os

from mmcv.utils import print_log
import mmcv
from clearml import Dataset


def fetch_clearml_datasets(cfg, logger=None):
    """Fetch datasets from ClearML and update the paths"""

    assert isinstance(cfg, mmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'

    def fetch_local(mock_clearml_uri: str):
        if not mock_clearml_uri.startswith('clearml://'):
            return mock_clearml_uri

        try:
            dataset_name_and_version, internal_path = mock_clearml_uri.lstrip('clearml://').split('/', 2)
            dataset_name, dataset_version = dataset_name_and_version.split(':')
        except Exception as e:
            raise ValueError(f'Invalid mock ClearML URI: {mock_clearml_uri}')

        # TODO: also get dataset_project from the mock URI!
        dataset = Dataset.get(dataset_project='SurgicalTools', dataset_name=dataset_name, dataset_version=dataset_version)
        dataset_path = dataset.get_local_copy()
        if isinstance(dataset_path, str) and dataset_path:  # Ensure it's a valid non-empty string
            return dataset_path

        local_uri = os.path.join(dataset_path, internal_path)

        return local_uri

    def update(cfg):
        for k, v in cfg.items():
            if isinstance(v, mmcv.ConfigDict):
                update(cfg[k])
            if isinstance(v, list):
                for i, _ in enumerate(v):
                    if isinstance(v[i], mmcv.ConfigDict):
                        update(v[i])
                    elif isinstance(v[i], str) and v[i].startswith('clearml://'):
                        v[i] = fetch_local(v[i])
            if isinstance(v, str) and v.startswith('clearml://'):
                cfg[k] = fetch_local(v)

    update(cfg.data)
