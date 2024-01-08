# Импорт библиотек
import io
import streamlit as st
from transformers import pipeline


st.title('Тест на токсика')


# Обученая модель для распознавания "токсичности" в тексте
clf = pipeline(
    task = 'sentiment-analysis', 
    model = 'SkolkovoInstitute/russian_toxicity_classifier')

text = [st.text_input('Введите ваш текст', '')]


def data(text):
    for row in text:
        yield row

for out in clf(data(text)):
    print(st.write(out))

#вывод
#{'label': 'neutral', 'score': 0.9872767329216003}
#{'label': 'toxic', 'score': 0.985331654548645}
