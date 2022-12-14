import torchvision.transforms as transforms
import torch

class Args:
    def __init__(self):
        self.e4e_model_path = "pretrained_models/e4e_ffhq_encode.pt"
        self.stylegan_model_path = "pretrained_models/stylegan2-ffhq-config-f.pt"
        self.shape_predictor_model_path = "pretrained_models/shape_predictor_68_face_landmarks.dat"
        self.resize_dims = (256, 256)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = None
        self.generator = None
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.image_input = None
        self.description = ''
        self.image = None
        self.latent = None
        self.method = None
        
