import streamlit as st
import torch
import numpy as np
from PIL import Image
from util.constants import IMAGE_INPUT, METHODS, MAPPER_PRETRAINED_MODELS
from encoder4editing.models.psp import pSp
from encoder4editing.utils.common import tensor2im
from encoder4editing.models.stylegan2.model import Generator
from argparse import Namespace
from util.latent_mapper_options import args
from runner import get_input_image_and_latent_code, get_edited_image
from encoder4editing.utils.common import tensor2im
from util.page_helper import display_img

@st.cache(allow_output_mutation=True)
def load_e4e():
    model_path = args.e4e_model_path
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['device'] = args.device
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cpu()
    return net

@st.cache(allow_output_mutation=True)
def load_stylegan_generator():
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(args.stylegan_model_path)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(args.device)
    return g_ema

@st.experimental_memo
def get_input_image():
    return get_input_image_and_latent_code(args)

def select_input_method():
    task_type = st.selectbox('Оберіть метод отримання вхідного зображення:', key='type_slbox', options=IMAGE_INPUT.keys())
    return task_type

def show_text_input(task_type):
    args.description = st.selectbox('Оберіть текстовий опис:', key='text_slbox', options=MAPPER_PRETRAINED_MODELS.keys())
    if (args.description is not None):
        selected_option = MAPPER_PRETRAINED_MODELS[args.description]
        args.checkpoint_path = selected_option["path"]
        args.no_coarse_mapper = selected_option["no_coarse_mapper"]
        args.no_medium_mapper = selected_option["no_medium_mapper"]
        args.no_fine_mapper = selected_option["no_fine_mapper"]
        solve_btn(task_type)

def latent_optimization():
    st.set_page_config(page_title="Метод латентного маперу", page_icon="📈")
    st.sidebar.title('Image Editing Service')
    st.header("Метод латентного маперу")
    args.method = METHODS["Latent Mapper"]
    args.net = load_e4e()
    args.generator = load_stylegan_generator()
    task_type = select_input_method()
    args.image_input = IMAGE_INPUT[task_type]
    if (task_type == "Завантажити зображення"):
        get_input_image.clear()
        file_buffer = st.file_uploader('', type=['jpeg', 'jpg', 'png'], accept_multiple_files=False)

        if file_buffer is not None:
            image = Image.open(file_buffer)
            img_original = np.array(image)
            args.image = img_original
            display_img(img_original)
            show_text_input(task_type)
    else:
        if(task_type == "Згенерувати зображення"):   
            generate_image_btn()
            input_image, latent = get_input_image()
            if (input_image is None):
                st.text("На вхідному зображенні не знайдено обличчя.")
                return
            args.image = input_image[0]
            args.latent = latent
            display_img(tensor2im(args.image))
            show_text_input(task_type)
                        

def solve_btn(task_type):
    if st.button('Редагувати', key='solve_btn'):
        col1, col2 = st.columns( [0.5, 0.5])
        if (task_type == "Завантажити зображення"):
            input_image, latent = get_input_image_and_latent_code(args)
            if (input_image is None):
                st.text("На вхідному зображенні не знайдено обличчя.")
                return
            args.image = input_image
            args.latent = latent
            with col1:
                st.markdown('<p>До</p>',unsafe_allow_html=True)
                display_img(tensor2im(input_image))
        else:
            if (task_type == "Згенерувати зображення"):
                with col1:
                    st.markdown('<p>До</p>',unsafe_allow_html=True)
                    display_img(tensor2im(args.image))
        final_image, loss_history = get_edited_image(args)
        with col2:
            st.markdown('<p>Після</p>',unsafe_allow_html=True)
            display_img(tensor2im(final_image))
        
        
def generate_image_btn():
    if st.button('Згенерувати нове зображення', key='generate_btn'):
        get_input_image.clear()
        
                           
if __name__ == '__main__':
    latent_optimization()
