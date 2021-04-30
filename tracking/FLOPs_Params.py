import _init_paths
from lib.models.models import LightTrackM_Speed

from thop import profile
from thop.utils import clever_format
import torch
# from efficientnet_pytorch.utils import Conv2dDynamicSamePadding
# from efficientnet_pytorch.utils import Conv2dStaticSamePadding
# from efficientnet_pytorch.utils import MemoryEfficientSwish
# from thop.vision.basic_hooks import count_convNd, zero_ops

if __name__ == "__main__":
    # Compute the Flops and Params of our LightTrack-Mobile model
    # build the searched model
    path_name = 'back_04502514044521042540+cls_211000022+reg_100000111_ops_32'  # LightTrack-Mobile model
    model = LightTrackM_Speed(path_name=path_name)
    print(model)
    backbone = model.features
    head = model.head

    x = torch.randn(1, 3, 256, 256)
    zf = torch.randn(1, 96, 8, 8)

    inp = {'cls': torch.randn(1, 128, 16, 16), 'reg': torch.randn(1, 128, 16, 16)}
    oup = model(x, zf)

    # custom_ops = {
    #     Conv2dDynamicSamePadding: count_convNd,
    #     Conv2dStaticSamePadding: count_convNd,
    #     MemoryEfficientSwish: zero_ops,
    # }
    # compute FLOPs and Params
    # the whole model
    macs, params = profile(model, inputs=(x, zf), custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)
    # backbone
    macs, params = profile(backbone, inputs=(x,), custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('backbone macs is ', macs)
    print('backbone params is ', params)
    # head
    macs, params = profile(head, inputs=(inp,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('head macs is ', macs)
    print('head params is ', params)
