_base_ = [
    'mask_rcnn_r50_fpn.py',
    'coco_instance.py',
    'schedule_2x.py',
    'default_runtime.py'
]
# FasterRcnn参数解释：https://blog.csdn.net/weicao1990/article/details/91879513


################################## 实验0517 ##################################
'''
roi_head.bbox_head.num_classes = 6 => 7 # 不对，这个不用改，undefined就是代表“背景”
roi_head.bbox_roi_extractor.roi_layer.sampling_ratio = 0 => 128
train_pipeline取消Resize，添加
dict(type='RandomCrop', crop_size=(1024, 1024), crop_type='absolute'),
test_pipeline取消img_scale
添加scale_factor = 1.0与RandomCrop
'''

################################## 实验0519 ##################################
'''
重新理解了一下pipline的概念，似乎resize是不可或缺的
在train_pipeline和test_pipeline上又添加了img_scale
'''

# mask-rcnn 训练步骤：https://blog.csdn.net/whunamikey/article/details/109847762

'''
# tensorflow-maskrcnn 参数
# Train on 1 GPU and 8 images per GPU. We can put multiple images on each
# GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
GPU_COUNT = 1
IMAGES_PER_GPU = 4

# Number of classes (including background)
NUM_CLASSES = 1 + 6  # background + 3 shapes

# Use small images for faster training. Set the limits of the small side
# the large side, and that determines the image shape.
IMAGE_RESIZE_MODE = 'none' # 'none','square','crop'
IMAGE_MIN_DIM = 1024
IMAGE_MAX_DIM = 1024
IMAGE_MIN_SCALE = 0

 # Backbone network architecture
# Supported values are: resnet50, resnet101.
# You can also provide a callable that should have the signature
# of model.resnet_graph. If you do so, you need to supply a callable
# to COMPUTE_BACKBONE_SHAPE as well
BACKBONE = "resnet50"

# Minimum probability value to accept a detected instance
# ROIs below this threshold are skipped    
DETECTION_MIN_CONFIDENCE = 0.5

# Non-maximum suppression threshold for detection
DETECTION_NMS_THRESHOLD = 0.3

# Image mean (RGB)
#MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
MEAN_PIXEL = np.array([139.5, 102.6, 86.4])
#MEAN_PIXEL = np.array([157.8, 116.7, 98.5])
# MEAN_STD = np.array([78.6, 62.5, 56.2])

# Use smaller anchors because our image and objects are small
RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

# Reduce training ROIs per image because the images are small and have
# few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
TRAIN_ROIS_PER_IMAGE = 128

# Use a small epoch since the data is simple
STEPS_PER_EPOCH = 50

# use small validation steps since the epoch is small
VALIDATION_STEPS = 3
'''