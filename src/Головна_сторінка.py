import streamlit as st
from helper.markdown_helper import markdown_to_string


if __name__ == '__main__':
    st.set_page_config(page_title="Сервіс з редагування зображення за текстовим описом", page_icon="📊")
    st.sidebar.title('Image Editing Service')
    st.header("Сервіс з редагування зображення за текстовим описом")
    st.markdown(markdown_to_string('markdown_data/general_page.md'))