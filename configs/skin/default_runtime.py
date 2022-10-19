checkpoint_config = dict(interval=1) # 每1个epoch存储一次模型
# yapf:disable
log_config = dict(
    interval=20, # 每50个batch输出一次信息
    hooks=[
        dict(type='TextLoggerHook'), # 控制台输出信息的风格
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl') # 分布式参数
log_level = 'INFO' # 输出信息的完整度级别
load_from = None # 加载模型的路径，None表示从预训练模型加载
resume_from = None # 恢复训练模型的路径
workflow = [('train', 1)] # 当前工作区名称
