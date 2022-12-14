from util.options import Args

class LatentMapperArgs(Args):
    def __init__(self):
        super().__init__()
        self.checkpoint_path = ''
        self.no_coarse_mapper = False
        self.no_medium_mapper = False
        self.no_fine_mapper = False
        
        
args = LatentMapperArgs()
