import streamlit as st
import plotly.graph_objects as go
import glob
import moviepy.editor as mpy
import pybase64
from util.helper import remove_files

def display_img(img):
    st.image(
                img,
                width=256,
            )
    

def loss_plot(loss_history, xlabel, ylabel, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(loss_history)+1)), y=loss_history,
                        opacity=1,
                        line=dict(color='firebrick', width=1),
                        mode='lines+markers'))
    fig.update_layout(title=title, autosize=False,
                      width=800,
                      height=600,
                      xaxis_title=xlabel,
                      yaxis_title=ylabel)
    st.plotly_chart(fig)
    
def create_gif(results_dir):
    gif_name = 'final_result'
    file_list = sorted(glob.glob(results_dir + '/*'))
    clip = mpy.ImageSequenceClip(file_list, fps=3)
    gif_path = 'final_result/{}.gif'.format(gif_name)
    clip.write_gif(gif_path, fps=3)
    file_ = open(gif_path, "rb")
    contents = file_.read()
    data_url = pybase64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'''
        <p>Візуалізація ітеративних результатів</p>
        <img src="data:image/gif;base64,{data_url}" alt="Зміна картинки" style="width: 256px; height: 256px;">''',
        unsafe_allow_html=True,
    )
    remove_files(file_list)
