from flask import Flask
from flask import request
from flask import render_template

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


with open('intents3.json', encoding='utf-8') as file:
  data = json.load(file)

with open('data.pickle', 'rb') as f:
    words, labels, training, output = pickle.load(f)



tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)

model.load('model.tflearn')


def bag_of_words(s, words):
  bag = [0 for _ in range(len(words))]

  s_words = nltk.word_tokenize(s)
  s_words = [stemmer.stem(w.lower()) for w in s_words]

  for se in s_words:
    for i, w in enumerate(words):
      if w == se:
        bag[i] = 1

  return numpy.array(bag)

app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     test = ('Hello nice World Page\n')
#     return f'Hello, World! {test}'

@app.route('/')
def home():
    return render_template('home.html')
  
@app.route('/predict', methods=['POST']) 
def make_prediction():
  newData = request.form.get('question')
  newDataLow = newData.lower()

  results = model.predict([bag_of_words(newDataLow, words)])[0]
  results_index = numpy.argmax(results)
  tag = labels[results_index]

  if results[results_index] > .50:
    for tg in data['intents']:
      if tg['tag'] == tag:
        responses = tg['responses']  
    answer =  random.choice(responses)    
    # return random.choice(responses)
    return render_template('home.html', answer=answer, question=newData)
  else:
    resp = [
        "Sorry, can't understand you",
        "Please give me more info",
        "Not sure I understand"
      ]
    answer =  random.choice(resp)
    return render_template('home.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=80)