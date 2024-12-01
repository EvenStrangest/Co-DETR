import os

from mmcv.utils import print_log
import mmcv
from clearml import Dataset


def fetch_clearml_datasets(cfg, logger=None):
    """Fetch datasets from ClearML and update the paths
    Format of mock ClearML URI: clearml://<project>/<dataset_name>:<dataset_version>/<internal_path>
    """

    assert isinstance(cfg, mmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'

    def fetch_local(mock_clearml_uri: str):
        if not mock_clearml_uri.startswith('clearml://'):
            return mock_clearml_uri

        try:
            project_name, dataset_name_and_version, internal_path = mock_clearml_uri.lstrip('clearml://').split('/', 3)
            dataset_name, dataset_version = dataset_name_and_version.split(':')
        except Exception as e:
            raise ValueError(f'Invalid mock ClearML URI: {mock_clearml_uri}')

        # TODO: cache the call to Dataset.get
        dataset = Dataset.get(dataset_project=project_name, dataset_name=dataset_name, dataset_version=dataset_version)
        dataset_path = dataset.get_local_copy()
        if not isinstance(dataset_path, str):
            raise ValueError(f'Failed to fetch dataset: {dataset_name} version: {dataset_version} from project: {project_name}')

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
