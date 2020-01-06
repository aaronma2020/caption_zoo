''' the configure of soft attention'''
''' soft attention 的配置文件'''
from .nic import NIC_cfg

class SoftAtt_cfg(NIC_cfg):

    def __init__(self):

        super(SoftAtt_cfg, self).__init__()
        self.model = 'att'
        self.fea_dim = 512
        self.att_dim = 100
        self.batch_size = 64
        self.lam = 1 # attention regular loss rate (attention正则损失比重）
        self.en_lr = 1e-4
        self.vis_dir = '../../eval_log/{}/{}/vis_img/'


