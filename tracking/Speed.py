import _init_paths
from lib.models.models import LightTrackM_Speed
import torch
import time

if __name__ == "__main__":
    # test the running speed
    path_name = 'back_04502514044521042540+cls_211000022+reg_100000111_ops_32'  # our 530M model
    use_gpu = True
    torch.cuda.set_device(0)
    model = LightTrackM_Speed(path_name=path_name)
    x = torch.randn(1, 3, 256, 256)
    zf = torch.randn(1, 96, 8, 8)
    if use_gpu:
        model = model.cuda()
        x = x.cuda()
        zf = zf.cuda()
    # oup = model(x, zf)

    T_w = 10  # warmup
    T_t = 100  # test
    with torch.no_grad():
        for i in range(T_w):
            oup = model(x, zf)
        t_s = time.time()
        for i in range(T_t):
            oup = model(x, zf)
        t_e = time.time()
        print('speed: %.2f FPS' % (T_t / (t_e - t_s)))
