import os
import streamlit as st
from argparse import Namespace
import torch
from latent_mapper.styleclip_mapper import StyleCLIPMapper
from datetime import datetime

@st.cache(allow_output_mutation=True)
def load_mapper(opts):
    net = StyleCLIPMapper(opts)
    net.eval()
    net.to(opts.device)
    return net
    
def latent_mapper_inference(args):
    start_time = datetime.now()
    # update test options with options used during training
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(args))
    opts = Namespace(**opts)
    net = load_mapper(opts)
    with torch.no_grad():
        latent = opts.latent.to(opts.device)
        result_batch = run_on_batch(latent, net)
        end_time = datetime.now()
        st.text('Час виконання: {}'.format(end_time - start_time))
        return result_batch[0][0], result_batch[1][0]


def run_on_batch(w, net):
	with torch.no_grad():
		w_hat = w + 0.1 * net.mapper(w)
		x_hat, w_hat = net.decoder([w_hat], input_is_latent=True, return_latents=True,
											randomize_noise=False, truncation=1)
		result_batch = (x_hat, w_hat)
	return result_batch

