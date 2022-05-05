
import argparse, os, shutil
import os.path as osp
from tqdm import tqdm
import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.evaluation import eval_map
from mmdet.core.visualization import imshow_gt_det_bboxes
from mmdet.datasets import build_dataset, get_loading_pipeline, CocoDataset


def bbox_map_eval(det_result, annotation):
    """Evaluate mAP of single image det result.

    Args:
        det_result (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotation (dict): Ground truth annotations where keys of
             annotations are:

            - bboxes: numpy array of shape (n, 4)
            - labels: numpy array of shape (n, )
            - bboxes_ignore (optional): numpy array of shape (k, 4)
            - labels_ignore (optional): numpy array of shape (k, )

    Returns:
        float: mAP
    """

    # use only bbox det result
    if isinstance(det_result, tuple):
        bbox_det_result = [det_result[0]] # bbox_det_result
    else:
        bbox_det_result = [det_result]
    # mAP
    iou_thrs = np.linspace(
        .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    mean_aps = []

    eval_results_thrs = []
    for thr in iou_thrs:
        # 每个图片每个类别的map
        mean_ap, eval_results = eval_map(bbox_det_result, [annotation], iou_thr=thr, logger='silent')

        #print('mean_ap:', mean_ap)
        #print('bbox_det_result:', type(bbox_det_result), len(bbox_det_result), bbox_det_result[0])
        # print('annotation:', annotation)

        mean_aps.append(mean_ap)
        eval_results_thrs.append(eval_results)
        break

    return sum(mean_aps) / len(mean_aps), eval_results_thrs  # 对不同的iou阈值再算一次map


class ResultVisualizer:
    """Display and save evaluation results.

    Args:
        show (bool): Whether to show the image. Default: True
        wait_time (float): Value of waitKey param. Default: 0.
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0
    """

    def __init__(self, show=False, wait_time=0, score_thr=0.5):
        self.show = show
        self.wait_time = wait_time
        self.score_thr = score_thr

    def _save_image_gts_results(self, dataset, results, out_dir=None):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        if not os.path.exists(out_dir):
            mmcv.mkdir_or_exist(out_dir)


        for index in tqdm(range(len(results))):
            data_info = dataset.prepare_train_img(index)

            # calc save file path
            filename = data_info['filename']
            if data_info['img_prefix'] is not None:
                filename = osp.join(data_info['img_prefix'], filename)
            else:
                filename = data_info['filename']
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname + '_' + name
            out_file = osp.join(out_dir, save_filename)
            imshow_gt_det_bboxes(
                data_info['img'],
                data_info,
                results[index],
                dataset.CLASSES,
                show=self.show,
                score_thr=self.score_thr,
                wait_time=self.wait_time,
                out_file=out_file)

    def evaluate_and_show(self,
                          dataset,
                          results,
                          topk=20,
                          show_dir='work_dir',
                          eval_fn=None):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Det results from test results pkl file
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
            eval_fn (callable, optional): Eval function, Default: None
        """
        # bbox_map_eval
        all_dir = osp.abspath(osp.join(show_dir, 'all'))
        self._save_image_gts_results(dataset, results, all_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=0,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--topk',
        default=800,
        type=int,
        help='saved Number of the highest topk '
             'and lowest topk after index sorting')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.5,
        help='score threshold (default: 0.)')
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
    args = parser.parse_args(args=['../configs/skin/skin_config.py', #  预测大图时，要修改mask_rcnn_r50_fpn.py test_cfg
                                   #'./output_pkldir/outputs_test_whole.pkl',
                                   '../output_pkldir/outputs_testwhole.pkl',
                                   '../results_analysis_show',
                                   '--eval', 'bbox'])
    return args



args = parse_args()

mmcv.check_file_exist(args.prediction_path)

cfg = Config.fromfile(args.config)
if args.cfg_options is not None:
    cfg.merge_from_dict(args.cfg_options)
cfg.data.test.test_mode = True
# import modules from string list.
if cfg.get('custom_imports', None):
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**cfg['custom_imports'])

cfg.data.test.pop('samples_per_gpu', 4)
# cfg.data.test.samples_per_gpu = 4
cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
dataset = build_dataset(cfg.data.test)
outputs = mmcv.load(args.prediction_path)


result_visualizer = ResultVisualizer(args.show, args.wait_time, args.show_score_thr)
result_visualizer.evaluate_and_show(dataset, outputs, topk=args.topk, show_dir=args.show_dir)


