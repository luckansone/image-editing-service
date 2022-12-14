def get_input_image_and_latent_code(options):
    input_image, latent_code = options.image_input(options)
    return input_image, latent_code


def get_edited_image(options):
    final_image, loss_history = options.method(options)
    return final_image, loss_history
    