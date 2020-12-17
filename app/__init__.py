import os
import pathlib
from bs4 import BeautifulSoup

import tensorflow as tf
import streamlit as st

# from modules.utils import *

BATCH_SIZE = 128
TEXT_FILE = 'dataset/tunes_v2.txt'
VOCAB_FILE = 'dataset/vocab_v2.json'

# load dataset
# ds = TextDatasetLoader(sequence_length=143,
#                        batch_size=BATCH_SIZE,
#                        buffer_size=1,
#                        text_dir=TEXT_FILE,
#                        output_dir=VOCAB_FILE,
#                        dataset_size='full')
# data_small = ds.load_dataset()
# vocab = ds.load_vocab(VOCAB_FILE)
# vocab_length = len(vocab)
# char2idx = ds.char2index(vocab)
# idx2char = ds.index2char(vocab)

# load model
# model = tf.keras.models.load_model(os.path.join("model", "music_composer_jazz4_lstm"))

st.title("Deep Jazz Composer")

st.markdown('Hi and welcome to Deep Jazz Composer.')
st.markdown("""
My name is Shayan Dadman, and this web application is part of my master dissertation under 
the title of "Synthetic Music Composition" at the UiT-Arctic 
University of Norway, Narvik. Throughout my master dissertation, 
I have investigated the current state of the art methods in the field, 
including the essentials in Artificial Neural Networks, 
automatic music composition, and audio/signal processing techniques. 
Furthermore, I examined the related approaches for Music Analysis tasks 
such as pitch detection and chord recognition, Music Information Retrieval (MIR) 
tasks like genre and instrument classification, and automatic music composition.
""")

# model parameters
st.markdown("""
Here you can experience the automatic music composition model proposed in my 
master thesis work. The model has trained on the excerpt of 52 tunes from 
Jazz musicians like John Coltrane, Bill Evans, Duke Ellington, Miles Davis, 
Thelonious Monk, etc. During the training, it has learned the jazz conventions 
and musical expressions. To compose a piece of music, first, we begin by 
defining some of the model's parameters.
""")

# length
st.subheader('Input Length')
st.markdown("""
This parameter defines the length of the final transcription. 
Use different values and figure which one is closer to your desire composition.\n

I recommend to start with a value between 400 to 600.
""")
length = st.number_input('Input length', value=200, min_value=1, step=1)

# temperature
st.subheader('Model Temperature')
st.markdown("""
Temprature value determines the consistency of the model. 
The values around one make the model predictive.
Indeed, it creates more robust transcriptions regarding the training examples.\n

However, by increasing the temperature value, we let the model's creativity to run wild. 
In other words, the model is less constrained by musical conventions in training examples.\n

My suggestion is to start with values around 0.5 and 0.7. Then turn it up ;)
""")
temp = st.number_input('Model temperature', value=0.5, min_value=0.1, step=0.1)

# composition header
st.subheader('Composition Properties')
st.markdown("""
It is possible to ask for composition by specifying musical properties 
such as tempo, length of the note, the measure, and key of the piece.\n

Below, you can define these properties or just use the default values.
""")
note_length = st.text_input(label='Note length', value='1/8')
measure = st.text_input(label='Measure', value='4/4')
tempo = st.text_input(label='Tempo', value='120')
key = st.text_input(label='Key', value='Eb')

# text box for model input
st.markdown("""
Very well. Now our model is all set and ready. 
Additionally, We can initialize the model with a start string. 
The start string can be a single chord from the progression or sequence of notes from the melody. 
For instance, you can give the "F6" chord, a short melody like "B2_c2" or a combination of both like ""Dm7" AGAG-". 
Remember, you may get different results based on the provided start string.
""")

input_string = st.text_input(label='Start string', value='')

st.markdown('Now, press the button, sit back and see what it will compose.')
# concatenate the composition props with input string
input_sample = "M:{m}\nL:{l}\nQ:{q}\nK:{k}\n".format(m=measure, l=note_length, q=tempo, k=key) + input_string
# if button clicked, compose
if st.button("Jazz me up!"):
    music = composer(model, start_string=input_sample, length=length, temperature=temp, encoder=char2idx, decoder=idx2char)
    music = "X:1\nC:Deep Jazz Composition\n"+ music
    st.text(music)


# st.markdown('Now ')
# HtmlFile = open('https://www.abcjs.net/abcjs-editor.html', 'r', encoding='utf-8')
# source_code = HtmlFile.read() 
# print(source_code)
# components.html(source_code, height=600, scrolling=True)

# JS = """<script src="modules/abcjs_plugin.js" type="text/javascript"></script>"""

# # Insert the script in the head tag of the static template inside your virtual environement
# index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"

# soup = BeautifulSoup(index_path.read_text(), "lxml")
# if not soup.find(id='abcjs-plugin'):
#     script_tag = soup.new_tag("script", id='abcjs-plugin')
#     script_tag.string = JS
#     soup.head.append(script_tag)
#     index_path.write_text(str(soup))