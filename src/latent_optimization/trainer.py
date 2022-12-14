import clip
import torch
import streamlit as st
import os
from torch import optim
from stqdm import stqdm
import torchvision
from util.helper import get_lr
from util.early_stopping import EarlyStopping
from datetime import datetime
    

def latent_optimization_trainer(args):
    os.makedirs(args.results_dir, exist_ok=True)
    text_inputs = torch.cat([clip.tokenize(args.description)]).to(args.device)

    # Initialize the latent vector to be updated.
    latent = args.latent.detach().clone()
    latent.requires_grad = True

    clip_loss = args.clip_loss
    optimizer = optim.Adam([latent], lr=args.lr)
    loss_history = []
    g_ema = args.generator
    early_stopping = EarlyStopping(args.early_stopping_min_dif)
    prev_val = 0
    start_time = datetime.now()

    for i in stqdm(range(args.step)):
        # Adjust the learning rate.
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        # Generate an image using the latent vector.
        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)

        # Calculate the loss value.
        c_loss = clip_loss(img_gen, text_inputs)
        l2_loss = ((args.latent - latent) ** 2).sum()
        loss = c_loss + args.l2_lambda * l2_loss

        # Get gradient and update the latent vector.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if args.save_intermediate_image and args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
            torchvision.utils.save_image(img_gen, f"results/{str(i).zfill(5)}.png", normalize=True, range=(-1, 1))
            
        if (early_stopping.early_stop(prev_val, loss)):
            break
        prev_val = loss

    end_time = datetime.now()
    st.text('Час виконання: {}'.format(end_time - start_time))
    return img_gen, loss_history
