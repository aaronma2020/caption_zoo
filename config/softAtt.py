''' the configure of NIC'''
''' NIC的配置文件'''
from .nic import NIC_cfg

class SoftAtt_cfg(NIC_cfg):

    def __init__(self):

        super(SoftAtt_cfg, self).__init__()

        self.model = 'att'
        self.fea_dim = 512
        self.att_dim = 100
        self.batch_size = 64
        self.lam = 1e-5 # attention regular loss rate (attention正则损失比重）
