from losses.clip_loss import CLIPLoss
from util.options import Args

class LatentOptimizationArgs(Args):
    def __init__(self):
        super().__init__()
        self.lr_rampup = 0.05
        self.lr = 0.1
        self.step = 20
        self.l2_lambda = 0.005
        self.save_intermediate_image_every = 1
        self.save_intermediate_image = True
        self.results_dir = "results"
        self.clip_loss = CLIPLoss(self.device)
        self.early_stopping_min_dif = 0.001
        
        
args = LatentOptimizationArgs()
