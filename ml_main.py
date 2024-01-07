# Импорт библиотек
import io
import streamlit as st
from transformers import pipeline

import streamlit as st
import pandas as pd

st.title('Тест на токсика')


text = st.text_input('Enter your text', '')


# Обученая модель для распознавания "токсичности" в тексте
clf = pipeline(
    task = 'sentiment-analysis', 
    model = 'SkolkovoInstitute/russian_toxicity_classifier')



def data(text):
    for row in text:
        yield row

for out in clf(data(text)):
    print(st.write(out))

#вывод
#{'label': 'neutral', 'score': 0.9872767329216003}
#{'label': 'toxic', 'score': 0.985331654548645}