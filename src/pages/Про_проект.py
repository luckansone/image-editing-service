import streamlit as st
from helper.markdown_helper import markdown_to_string

def about_us_page():
    st.set_page_config(page_title="Про проект")
    st.sidebar.title('Image Editing Service')
    st.markdown(markdown_to_string('markdown_data/about_us.md'))
    

if __name__ == '__main__':
    about_us_page()