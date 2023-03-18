import streamlit as st
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import sklearn

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Spam Classifier')

ps=PorterStemmer()

input_sms=st.text_input('Enter the message')

def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text=y[:]
  y=[]
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text=y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

if st.button('Predict'):
    # 1)preprocess
    transformed=transform_text(input_sms)
    # 2)vectorize
    vector_input=tfidf.transform([transformed])
    # 3)predict
    result=model.predict(vector_input)[0]
    # 4)display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')