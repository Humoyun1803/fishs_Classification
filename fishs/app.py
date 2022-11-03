import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform
plt = platform.system()
if plt == 'Linux':pathlib.WindowsPath = pathlib.PosixPath

st.title("Dengiz hayvonlarini klassifikatsiya qiluvchi model")

#rasimni joylash
file = st.file_uploader('Rasm yuklash', type=['png','jpg','gif','svg'])
#model
if file:
    st.image(file)

    #PIL convert
    img = PILImage.create(file)


    model = load_learner('fishs_model.pkl')
    #predict

    pred, pred_id, probs = model.predict(img)

    st.success(f"Bashorat:{pred}")
    st.info(f'Ehtimollik:{probs[pred_id]*100 : .1f} %')
    fig = px.bar(x = probs*100, y = model.dls.vocab)
    st.plotly_chart(fig)


