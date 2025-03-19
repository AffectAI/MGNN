import torch
from mmengine.registry import MODELS
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp


model = dict(
    type='Recognizer2DGNN',
    backbone=dict(
        type='ResNet',
        pretrained='./work_dirs/pertrained_models/resnet50_ft_weight.pth',  # pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        depth=50,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN2d', requires_grad=True),  # dict(type='BN2d', requires_grad=True)
        norm_eval=False),
    neck=dict(
        type='TemporalSpatialFusion',
        in_channels=(256, 512, 1024, 2048),
        out_channels=256,
        temporal_choose='last_frame',  # 'mean' or 'last_frame'
        conv_cfg=dict(type='Conv2d'),
        norm_cfg=dict(type='GN', num_groups=32),
        aux_head_cfg=dict(
            dropout_ratio=0.4,
            loss_weight=0.5,
            loss_cls=dict(
                type='RegressionLoss',
                loss_type='BMC+MSE',
                loss_weight=[1, 1]))),
    cls_head=dict(
        type='AuxHead2',
        num_classes=1,
        in_channels=256,
        is_aux=False,
        dropout_ratio=0.4,
        conv_cfg=dict(type='Conv2d'),
        norm_cfg=dict(type='GN', num_groups=32),
        loss_cls=dict(
            type='RegressionLoss',
            loss_type='BMC+MSE',
            loss_weight=[1, 1])),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)


model = MODELS.build(model)
model.to('cuda')
model.load_state_dict(torch.load('/opt/data/private/mmaction2/work_dirs/exp2/Final_2014_rmse_6.96_mae_5.65.pth', map_location='cuda'))
model.eval()


img = read_image("/opt/data/private/Depression-detect-ResNet/data/new_processed/test/Northwind/249_1_Northwind_video/0000.jpg")
input_tensor = torch.from_numpy(resize(img, (224, 224)))
# input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
with SmoothGradCAMpp(model) as cam_extractor:
  # Preprocess your data and feed it to the model
  out = model(input_tensor.unsqueeze(0))
  # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)


import matplotlib.pyplot as plt
# Visualize the raw CAM
plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()




