from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class AddAugmentationID(object):
    """Custom pipeline component to add augmentation identifiers to image metadata."""

    def __call__(self, results):
        # Handle batch processing by checking if 'img_metas' is a list
        if isinstance(results['img_metas'], list):
            for img_meta in results['img_metas']:
                self._add_aug_id(img_meta)
        else:
            self._add_aug_id(results['img_metas'])
        return results

    def _add_aug_id(self, img_meta):
        ori_filename = img_meta.get('ori_filename', 'noname')
        aug_suffix = ''

        # Flip information
        if img_meta.get('flip', False):
            aug_suffix += f"_flip_{img_meta['flip_direction']}"
        else:
            aug_suffix += "_noflip"

        # Scale factor
        scale_factor = img_meta.get('scale_factor', 1.0)
        if isinstance(scale_factor, float):
            scale_factor = (scale_factor, scale_factor, scale_factor, scale_factor)
        aug_suffix += f"_scale_{scale_factor[0]:.2f}_{scale_factor[1]:.2f}"

        # Update the filename in img_meta
        img_meta['ori_filename'] = ori_filename + aug_suffix

    def __repr__(self):
        return self.__class__.__name__ + '()'
