
import argparse
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

    def _save_image_gts_results(self, dataset, results, mAPs, out_dir=None):
        mmcv.mkdir_or_exist(out_dir)

        for mAP_info in mAPs:
            index, mAP = mAP_info
            data_info = dataset.prepare_train_img(index)

            # calc save file path
            filename = data_info['filename']
            if data_info['img_prefix'] is not None:
                filename = osp.join(data_info['img_prefix'], filename)
            else:
                filename = data_info['filename']
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname + '_' + str(round(mAP, 3)) + name
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
                          eval_fn=None,
                          nosave=False):
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

        assert topk > 0
        if (topk * 2) > len(dataset):
            topk = len(dataset) // 2

        # bbox_map_eval
        if eval_fn is None:
            eval_fn = bbox_map_eval
        else:
            assert callable(eval_fn)

        prog_bar = mmcv.ProgressBar(len(results))
        _mAPs = {}

        eval_results_thrs_imgs = []
        num = 0
        for i, (result, ) in enumerate(tqdm(zip(results[:2]))): # for i, result in enumerate(tqdm(results))
            # self.dataset[i] should not call directly
            # because there is a risk of mismatch
            data_info = dataset.prepare_train_img(i) # category_id: 1~10
            mAP, eval_results_thrs = eval_fn(result, data_info['ann_info']) # 针对每个图片的map, 然后需要再根据图片再算一次


            _mAPs[i] = mAP
            prog_bar.update()

            eval_results_thrs_imgs.append(eval_results_thrs)

            num += 1
            # if num >= 2:
            #    break


        # 依次取出所有的图片检测结果
        num_class = len(results[0][0]) - 1
        class_tp_dict = {}
        class_fp_dict = {}
        class_gt_dict = {}
        class_ap_dict = {}
        all_tp = 0
        all_fp = 0
        all_gt = 0
        # all_ap_0 = 0
        #初始化默认为0
        for class_index in range(num_class):
            class_tp_dict[class_index] = 0
            class_fp_dict[class_index] = 0
            class_gt_dict[class_index] = 0
            class_ap_dict[class_index] = 0

        # 遍历图像
        for img_index, (eval_results_thrs, ) in enumerate(tqdm(zip(eval_results_thrs_imgs))):
            #print('img_index, eval_results_thrs:', img_index, eval_results_thrs)
            eval_results_IOU50 =  eval_results_thrs[0]

            # 遍历类型
            for class_index in range(num_class):
                eval_results_IOU50_class = eval_results_IOU50[class_index]
                class_tp_dict[class_index] += eval_results_IOU50_class['tp_50']
                all_tp +=eval_results_IOU50_class['tp_50']
                class_fp_dict[class_index] += eval_results_IOU50_class['fp_50']
                all_fp +=eval_results_IOU50_class['fp_50']
                class_gt_dict[class_index] += eval_results_IOU50_class['num_gts']
                all_gt +=eval_results_IOU50_class['num_gts']

                if eval_results_IOU50_class['num_gts'] > 0:
                    class_ap_dict[class_index] += eval_results_IOU50_class['ap']


        #for index, mAP in _mAPs.items():
        #    all_ap_0 += mAP
        #all_ap_0 = all_ap_0 / len(eval_results_thrs_imgs)

        eps = np.finfo(np.float32).eps
        precision_all = all_tp / np.maximum((all_tp + all_fp), eps)
        recall_all = all_tp / np.maximum(all_gt, eps)

        print('\nprecision_all, recall_all:', np.round(precision_all, 3), np.round(recall_all, 3)) # , all_ap_0
        for class_index in range(num_class):
            tmp_tp = class_tp_dict[class_index]
            tmp_fp = class_fp_dict[class_index]
            tmp_gt = class_gt_dict[class_index]
            #tmp_ap = class_ap_dict[class_index] / len(eval_results_thrs_imgs)
            precision_class = tmp_tp / np.maximum((tmp_tp + tmp_fp), eps)
            recall_class = tmp_tp / np.maximum(tmp_gt, eps)
            print(CocoDataset.get_classes()[class_index], np.round(precision_class, 3), np.round(recall_class, 3))

        # descending select topk image
        _mAPs = list(sorted(_mAPs.items(), key=lambda kv: kv[1]))
        print('_mAPs:', _mAPs)

        # 如果不保存图片，只看指标的话
        if not nosave:
            good_mAPs = _mAPs[-topk:]
            bad_mAPs = _mAPs[:topk]

            good_dir = osp.abspath(osp.join(show_dir, 'good'))
            bad_dir = osp.abspath(osp.join(show_dir, 'bad'))
            self._save_image_gts_results(dataset, results, good_mAPs, good_dir)
            self._save_image_gts_results(dataset, results, bad_mAPs, bad_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')
    parser.add_argument('--nosave', action='store_true', help='show results')
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
    args = parser.parse_args(args=[
                                '../configs/skin/skin_config.py',
                                #'../work_dirs/skin_config_imgv2/skin_config.py',
                                '../output_pkldir/val_procedge.pkl',
                                '../results_analysis_show',
                                # '--nosave', # 不保存图片
                                '--eval', 'bbox'])
    return args



args = parse_args()
#print(args.nosave) # default false


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

# evaluate mAP
eval_kwargs = cfg.get('evaluation', {}).copy()
kwargs = {}
# hard-code way to remove EvalHook args
for key in [
        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        'rule'
]:
    eval_kwargs.pop(key, None)
eval_kwargs.update(dict(metric=args.eval, **kwargs))
metric = dataset.evaluate(outputs, **eval_kwargs) #classwise=True,  这个在配置文件中设置的 # getCatIds() 在getgtids 加了这个的，所有有效。
print(metric)


# result_visualizer = ResultVisualizer(args.show, args.wait_time, args.show_score_thr)
# result_visualizer.evaluate_and_show(dataset, outputs, topk=args.topk, show_dir=args.show_dir, nosave=args.nosave)
#

