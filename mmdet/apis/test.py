import os.path as osp
import pickle
import shutil
import tempfile
import time
import numpy as np

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    valid_rpn=False,
                    collect_cpu=False):
    model.eval()
    results = []

    other_infos = []
    dataset = data_loader.dataset
    #data_info = dataset.prepare_train_img(82)

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        #fortest20211106
        # if 'CXM__祁雄伟_痤疮_20190926131328000_斑点_crop11.jpg' not in data['img_metas'][0].data[0][0]['filename']:
        #     prog_bar.update()
        #     continue

        with torch.no_grad():
            if valid_rpn:
                rpn_bfnms_list, proposal_list, bbox_score_label_bfnms_list, result = model(return_loss=False, rescale=True, return_proposalist=True, **data)

                # 如果将用生成CPU
                if collect_cpu:
                    rpn_bfnms_list_cpu = []
                    proposal_list_cpu = []
                    bbox_score_label_bfnms_list_cpu = []
                    for i in range(len(result)):
                        rpn_bfnms_bboxes, rpn_bfnms_scores, rpn_bfnms_ids, rpn_bfpre_anchors, rpn_bfpre_scores, rpn_bfpre_bbox_preds = rpn_bfnms_list[i]
                        roi_bfnms_bboxes, roi_bfnms_scores, roi_bfnms_labels = bbox_score_label_bfnms_list[i]
                        rpn_bfnms_list_cpu.append([rpn_bfnms_bboxes.to('cpu'),
                                                   rpn_bfnms_scores.to('cpu'),
                                                   rpn_bfnms_ids.to('cpu'),
                                                   [j.to('cpu') for j in rpn_bfpre_anchors],

                                                   # len_2: rpn cls_scores nwd_scores
                                                   [
                                                       [j.to('cpu') for j in rpn_bfpre_scores[0]], # rpn cls_scores
                                                        [j.to('cpu') for j in rpn_bfpre_scores[1] if rpn_bfpre_scores[1] is not None] # rpn nwd_scores 20211212 orirpn 可能有bug
                                                   ],

                                                   [j.to('cpu') for j in rpn_bfpre_bbox_preds]])
                        proposal_list_cpu.append([proposal_list[i].to('cpu')])
                        bbox_score_label_bfnms_list_cpu.append([roi_bfnms_bboxes.to('cpu'),
                                                                roi_bfnms_scores.to('cpu'),
                                                                roi_bfnms_labels.to('cpu')])
                    rpn_bfnms_list = rpn_bfnms_list_cpu
                    proposal_list = proposal_list_cpu
                    bbox_score_label_bfnms_list = bbox_score_label_bfnms_list_cpu



                # other_info 主要用于RPN proposal的分析
                if isinstance(result[0], tuple):
                    other_info = [(rpn_bfnms_list[i], proposal_list[i], bbox_score_label_bfnms_list[i]) for i in range(len(result))]
                else:
                    other_info = [rpn_bfnms_list, proposal_list, bbox_score_label_bfnms_list]

                other_infos.extend(other_info)
            else:
                result = model(return_loss=False, rescale=True, **data) # , return_proposalist=True

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # if isinstance(result[0], tuple):
        #     result_item = []
        #     for bbox_results, mask_results in result:
        #         mask_results = encode_mask_results(mask_results)
        #         # 屏蔽 filter other类的预测
        #         for class_ind in [9]: # 7, 8, 9
        #             bbox_results[class_ind] = np.empty(shape=[0, 5])
        #             mask_results[class_ind] = []
        #         result_item.append((bbox_results, mask_results))
        #     results.extend(result_item)
        # else:
        #     result_item = []
        #     for bbox_results in result:
        #         for class_ind in [9]: # 7, 8, 9
        #             bbox_results[class_ind] = np.empty(shape=[0, 5])
        #         result_item.append(bbox_results)
        #     results.extend(result_item)

        # print(len(result)) # 1
        # print(len(result[0])) # 2
        #print(type(result[0])) # 2
        # #print(result[0][0])
        # print(len(result[0][0])) # 10
        # print(len(result[0][1])) # 2
        # print(len(result[0][1][0])) # 10
        # print(len(result[0][1][1]))  # 10
        # input()

        if isinstance(result[0], tuple):
            # 删除部分类的结果
            result_item = []
            for bbox_results, mask_results in result:
                mask_results = encode_mask_results(mask_results)
                if isinstance(mask_results, tuple):  # mask scoring
                    cls_segms, cls_mask_scores = mask_results
                    for class_ind in [9]:  #
                        bbox_results[class_ind] = np.empty(shape=[0, 5])
                        cls_segms[class_ind] = []
                        cls_mask_scores[class_ind] = []
                    result_item.append((bbox_results, (cls_segms, cls_mask_scores)))
                else:
                    # cls_segms = mask_results
                    for class_ind in [9]: #
                        bbox_results[class_ind] = np.empty(shape=[0, 5])
                        mask_results[class_ind] = ()
                    result_item.append((bbox_results, mask_results))
            results.extend(result_item)
        else:
            result_item = []
            for bbox_results in result:
                for class_ind in [9]: # 7, 8, 9
                    bbox_results[class_ind] = np.empty(shape=[0, 5])
                result_item.append(bbox_results)
            results.extend(result_item)

        for _ in range(batch_size):
            prog_bar.update()

        #break #fortest20211106
        # # 限制mask的大小
        # if isinstance(result[0], tuple):
        #     result_item = []
        #     for bbox_results, mask_results in result:
        #         for class_ind, mask_label in enumerate(mask_results):
        #             if len(mask_label) == 0:
        #                 continue
        #             label_filter = np.ones(len(mask_label))
        #             # 注意要倒序遍历
        #             for item_ind in range(len(mask_label) - 1, -1, -1):
        #                 item_area = np.sum(mask_label[item_ind])
        #                 # 如果mask的面积小于100
        #                 if item_area < 100:
        #                     label_filter[item_ind] = 0
        #                     mask_label.pop(item_ind)
        #             bbox_results[class_ind] = bbox_results[class_ind][label_filter > 0, :]
        #         mask_results = encode_mask_results(mask_results)
        #         result_item.append((bbox_results, mask_results))
        #     results.extend(result_item)
        # else:
        #     results.extend(result)



    if valid_rpn:
        return other_infos, results
    else:
        return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
