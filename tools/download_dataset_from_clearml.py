from clearml import Dataset


def get_valid_dataset_path():
    dataset_name, dataset_version = 'RobotA_with_Photomontage_Round5', '1.0.0'

    while True:
        try:
            robota = Dataset.get(dataset_project='SurgicalTools', dataset_name=dataset_name,
                                 dataset_version=dataset_version)
            dataset_path = robota.get_local_copy()
            if isinstance(dataset_path, str) and dataset_path:  # Ensure it's a valid non-empty string
                return dataset_path
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")


# Usage
dataset_path = get_valid_dataset_path()
print(f"Valid dataset path: {dataset_path}")
