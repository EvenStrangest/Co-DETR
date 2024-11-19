import clearml

import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from projects import *
import projects.datasets.robota  # TODO: replace this with the correct entry in the `custom_imports` field of the config file, a la `custom_imports = dict(imports=['projects.datasets.robota'], allow_failed_imports=False)`


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    if os.environ.get('CHOICE_DATASET') == 'RobotA':
        # set the project name
        task_name = 'EvalRA'
    elif os.environ.get('CHOICE_DATASET') == 'LabA':
        # set the project name
        task_name = 'EvalLA'
    elif os.environ.get('CHOICE_DATASET') == 'LabC':
        # set the project name
        task_name = 'EvalLC'
    elif os.environ.get('CHOICE_DATASET') == 'COCO':
        # set the project name
        task_name = 'EvalCOCO'
    else:
        # set the project name
        task_name = 'EvalUnknown'

    args = parse_args()

    checkpoint_nickname = args.checkpoint.split('FtRA_')[-1].\
        replace('https://files.clear.ml/', '').replace('.pth', '').replace('/models', '').replace('/', '_')
    task_name += f"_{checkpoint_nickname}"

    # create ClearML task
    task = clearml.Task.init(project_name='Co-DETR', task_name=task_name, task_type=clearml.Task.TaskTypes.inference,
                             deferred_init = False, )
    task.set_base_docker(docker_image='361432929675.dkr.ecr.us-east-1.amazonaws.com/trackimed/co_detr_manual:2024OCT06',
                         docker_arguments = f'--env CHOICE_DATASET={os.environ.get("CHOICE_DATASET")}',
                         docker_setup_bash_script = '')

    # execute the task remotely
    # task.execute_remotely(queue_name="default")

    # set environment variable for the dataset path
    override_checkpoint_classes = True  # TODO: eventually, once we train models with the new mapping, we can turn this off
    if os.environ.get('CHOICE_DATASET') == 'RobotA':
        print("Using RobotA dataset")
        # robota = clearml.Dataset.get(dataset_id='4de72c7d8fc9489fb3b1bc292b0fb0e7')
        robota = clearml.Dataset.get(dataset_project='SurgicalTools', dataset_name='RobotA', dataset_version='1.2.0')
        os.environ['MMDET_DATASETS'] = robota.get_local_copy() + '/'
    elif os.environ.get('CHOICE_DATASET') == 'RobotA1ofeach':
        os.environ['MMDET_DATASETS'] = '/data/'
    elif os.environ.get('CHOICE_DATASET') == 'LabA':
        print("Using LabA dataset")
        laba = clearml.Dataset.get(dataset_project='SurgicalTools', dataset_name='LabA', dataset_version='1.0.0')
        os.environ['MMDET_DATASETS'] = laba.get_local_copy() + '/'
    elif os.environ.get('CHOICE_DATASET') == 'LabC':
        print("Using LabC dataset")
        labc = clearml.Dataset.get(dataset_project='SurgicalTools', dataset_name='LabC', dataset_version='1.0.0')
        os.environ['MMDET_DATASETS'] = labc.get_local_copy() + '/'
    elif os.environ.get('CHOICE_DATASET') == 'COCO':
        print("Using MS COCO dataset")
        # mscoco = clearml.Dataset.get(dataset_id='eaeccf28c682478c9badb6d5c5700437')
        mscoco = clearml.Dataset.get(dataset_project='MS_COCO', dataset_name='MS_COCO_2017', dataset_version='1.0.0')
        os.environ['MMDET_DATASETS'] = mscoco.get_local_copy() + '/'
    else:
        print(f"Using dataset from {os.environ.get('MMDET_DATASETS')}")

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    print(f'Config:\n{cfg.pretty_text}')

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        # TODO: not enforcing test mode is required for allowing visualization of annotations; however, enforcing test mode is required for the evaluation to work
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        # TODO: not enforcing test mode is required for allowing visualization of annotations; however, enforcing test mode is required for the evaluation to work
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)  # TODO: this is a way to access other sets within a dataset
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if override_checkpoint_classes or 'CLASSES' not in checkpoint.get('meta', {}):
        model.CLASSES = dataset.CLASSES
    else:
        model.CLASSES = checkpoint['meta']['CLASSES']

    # TODO: export this to args !!!
    if args.show_dir is not None and not distributed:
        show_results_kwargs = dict(thickness = 4, font_size = 40,)
    else:
        show_results_kwargs = dict()

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr, show_results_kwargs)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))

    # upload contents of show_dir to ClearML
    if args.show_dir is not None and os.path.exists(args.show_dir):
        # for image_name in os.listdir(args.show_dir):
        #     image_path = os.path.join(args.show_dir, image_name)
        #     if os.path.isfile(image_path):  # Ensure it's a file
        #         task.logger.report_image("debug_samples", image_name, iteration=0, image=image_path)
        task.upload_artifact(name="show_dir", artifact_object=args.show_dir)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
            task.upload_artifact(name="outputs", artifact_object=args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)
                task.upload_artifact(name="evaluation_results", artifact_object=json_file)


if __name__ == '__main__':
    main()
