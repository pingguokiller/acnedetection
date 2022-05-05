# MaskRCNN
# Debug:
# ../configs/skin/skin_config.py --work_dir ../work_dirs/skin_config_debug
# train.sh:
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/skin/skin_config.py --work_dir ./work_dirs/trainnwd

# faster_rcnn
# Debug:
# ../configs/faster_rcnn/skin_faster_rcnn_r50_fpn_2x_coco.py.py --work_dir ./work_dirs/faster_rcnn
# train.sh:
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/faster_rcnn/skin_faster_rcnn_r50_fpn_2x_coco_nwd.py --work_dir ./work_dirs/faster_rcnn_nwd5

# cascade_rcnn
# train.sh:
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/cascade_rcnn/skin_cascade_mask_rcnn_r50_fpn_1x_coco.py --work_dir ./work_dirs/cascade_rcnn3



# FCOS
# train.sh:
#CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/fcos/skin_fcos_r50_caffe_fpn_gn-head_1x_coco.py --work_dir ./work_dirs/fcos2

# AutoAssign
# Debug:
# ../configs/autoassign/my_autoassign_r50_fpn_8x2_1x_coco.py --work_dir ./work_dirs/autoassign
# train.sh:
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/autoassign/his_autoassign_r50_fpn_8x2_1x_coco.py --work_dir ./work_dirs/autoassign2

# Mask score RCNN
# train.sh:
# CUDA_VISIBLE_DEVICES=0 python tools/train.py work_dirs/msrcnn/skin_ms_rcnn_r50_fpn_1x_coco.py --work_dir ./work_dirs/msrcnn3


# ATSS
# train.sh:
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/atss/skin_atss_r50_fpn_1x_coco.py --work_dir ./work_dirs/atss
