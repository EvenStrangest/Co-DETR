import os

from mmcv.runner import Hook, HOOKS
from clearml import Task


@HOOKS.register_module()
class ClearMLCheckpointHook(Hook):
    def after_save_checkpoint(self, runner):  # TODO: there's no such stage in the runner! fix this!
        # Get the path to the latest checkpoint file
        checkpoint_path = runner.meta['hook_msgs']['last_ckpt']

        # Ensure the checkpoint file exists
        if os.path.exists(checkpoint_path):
            # Get the current ClearML task
            task = Task.current_task()
            if task:
                # Upload the checkpoint as an artifact
                task.upload_artifact(
                    name=f'checkpoint_step_{runner.step}',
                    artifact_object=checkpoint_path
                )
                print(f"Uploaded checkpoint for epoch {runner.epoch + 1} to ClearML.")
            else:
                print("No active ClearML task found.")
        else:
            print(f"Checkpoint {checkpoint_path} does not exist.")

