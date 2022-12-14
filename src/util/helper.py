import torch
import dlib
import os
import math
from encoder4editing.utils.alignment import align_face


def run_alignment(image, model_path):
    predictor = dlib.shape_predictor(model_path)
    try:
        aligned_image = align_face(input_image=image, predictor=predictor)
    except:
        return None
    return aligned_image 

def run_on_batch(inputs, net, device):
    images, latents = net(inputs.to(device).float(), randomize_noise=False, return_latents=True)
    return images, latents

def transform_image_to_vector(args):
    input_image = run_alignment(args.image, args.shape_predictor_model_path)
    if input_image is None:
        return None, None
    input_image.resize(args.resize_dims)
    transformed_image = args.transform(input_image)
    with torch.no_grad():
        images, latents = run_on_batch(transformed_image.unsqueeze(0), args.net, args.device)
        img_orig, latent_code = images[0], latents[0]
    
    latent_code = torch.unsqueeze(latent_code, dim=0)
    return img_orig, latent_code

def generate_random_image(args):
    mean_latent = args.generator.mean_latent(4096)
    latent_code_init_not_trunc = torch.randn(1, 512).to(args.device)
    with torch.no_grad():
        image, latent_code = args.generator([latent_code_init_not_trunc], return_latents=True,
                                    truncation=0.7, truncation_latent=mean_latent)
    return image, latent_code


def get_lr(t, initial_lr, rampdown=0.50, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def string_is_not_empty (string):
    return bool(string and string.strip())

def remove_files(files):
    for f in files:
        os.remove(f)

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt   
