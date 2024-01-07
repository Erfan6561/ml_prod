# Импорт библиотеки
from transformers import pipeline

# Обученая модель для распознавания "токсичности" в тексте
clf = pipeline(
    task = 'sentiment-analysis', 
    model = 'SkolkovoInstitute/russian_toxicity_classifier')

# Сам текст, пример
text = ['Какой замечательный новый год!',
        'Я не собираюсь терпеть эту чушь с её стороны, уф.']

def data(text):
    for row in text:
        yield row

for out in clf(data(text)):
    print(out)

#вывод в виде
#{'label': 'neutral', 'score': 0.9872767329216003}
#{'label': 'toxic', 'score': 0.985331654548645}
