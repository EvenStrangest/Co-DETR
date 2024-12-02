from clearml import Dataset


def get_valid_dataset_path(project: str, dataset_name: str, dataset_version: str):

    tries_so_far = 0
    while True:
        try:
            tries_so_far += 1
            print(f"Attempt #{tries_so_far} to get the dataset path ...")
            robota = Dataset.get(dataset_project=project, dataset_name=dataset_name, dataset_version=dataset_version)
            dataset_path = robota.get_local_copy()
            if isinstance(dataset_path, str) and dataset_path:
                print(f"Dataset path: {dataset_path}")
                return dataset_path
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")


# Usage
dataset_path = get_valid_dataset_path('SurgicalTools', 'RobotA_with_Photomontage_Round5', '1.0.0')
