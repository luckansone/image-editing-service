import torch
from torch import nn
import latent_mapper.latent_mappers as latent_mapper
from encoder4editing.models.stylegan2.model import Generator
from util.helper import get_keys

class StyleCLIPMapper(nn.Module):
	def __init__(self, opts):
		super(StyleCLIPMapper, self).__init__()
		self.opts = opts
		# Define architecture
		self.mapper = self.set_mapper()
		self.decoder = Generator(1024, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d(opts.resize_dims)
		# Load weights if needed
		self.load_weights()

	def set_mapper(self):
		return latent_mapper.LevelsMapper(self.opts)

	def load_weights(self):
     # update test options with options used during training
		ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
		self.mapper.load_state_dict(get_keys(ckpt, 'mapper'), strict=True)
		self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.mapper(x)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images
