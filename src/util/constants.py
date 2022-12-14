from util.helper import transform_image_to_vector, generate_random_image
from latent_optimization.trainer import latent_optimization_trainer
from latent_mapper.inference import latent_mapper_inference

METHODS = {
    "Latent Optimization": latent_optimization_trainer,
    "Latent Mapper": latent_mapper_inference
}

IMAGE_INPUT = {
    "Завантажити зображення": transform_image_to_vector,
    "Згенерувати зображення": generate_random_image
}

MAPPER_PRETRAINED_MODELS = {
    "angry face":{
            "path": "mapper_pretrained_models/angry.pt",
            "no_coarse_mapper": False,
            "no_medium_mapper": False,
            "no_fine_mapper": True
    },
    "surprised face": {
        "path": "mapper_pretrained_models/surprised.pt",
        "no_coarse_mapper": False,
        "no_medium_mapper": False,
        "no_fine_mapper": True
    },
    "curly hair": {
        "path": "mapper_pretrained_models/curly_hair.pt",
        "no_coarse_mapper": False,
        "no_medium_mapper": False,
        "no_fine_mapper": True
    },
    "purple hair": {
        "path": "mapper_pretrained_models/purple_hair.pt",
        "no_coarse_mapper": False,
        "no_medium_mapper": False,
        "no_fine_mapper": False
    },
    "afro hair": {
            "path": "mapper_pretrained_models/afro.pt",
            "no_coarse_mapper": False,
            "no_medium_mapper": False,
            "no_fine_mapper": True
    },
    "Mark Zuckerberg": {
        "path": "mapper_pretrained_models/zuckerberg.pt",
        "no_coarse_mapper": False,
        "no_medium_mapper": False,
        "no_fine_mapper": False
    }
}