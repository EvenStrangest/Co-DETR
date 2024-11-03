from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class AddAugmentationID(object):
    """Custom pipeline component to add augmentation identifiers to image metadata."""

    def __call__(self, results):
        # TODO: may fail for batched data

        ori_filename = results.get('ori_filename', 'noname')
        aug_suffix = ''

        # Flip information
        if results.get('flip', False):
            aug_suffix += f"_flip_{results['flip_direction']}"
        else:
            aug_suffix += "_noflip"

        # Scale factor (if applicable)
        if 'scale_factor' in results:
            scale_factor = results['scale_factor']
            if isinstance(scale_factor, float):
                scale_factor = (scale_factor, scale_factor)
            aug_suffix += f"_scale_{scale_factor[0]:.2f}_{scale_factor[1]:.2f}"

        # Add other augmentation identifiers as needed
        # For example, rotation angle, color jitter parameters, etc.

        # Update the 'ori_filename' in results
        results['ori_filename'] = ori_filename + aug_suffix

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'
