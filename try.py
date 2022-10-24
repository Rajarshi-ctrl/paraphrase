# pip install torch
# pip install sentence-splitter
# pip install SentencePiece
# pip install transformers

import torch
import time
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_splitter import SentenceSplitter
from pynput.keyboard import Controller


model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences):
    batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

# text = "Graphical user interface, is a form of user interface that allows users to interact with electronic devices through graphical icons and audio indicator such as primary notation, instead of text-based UIs, typed command labels or text navigation."
# get_response(text, 3)

context = "In this video, I will be showing you how to build a stock price web application in Python using the Streamlit and yfinance library. The app will be able to retrieve company information as well as the stock price data for S and P 500 companies. All of this in less than 50 lines of code."

splitter = SentenceSplitter(language='en')
sentence_list = splitter.split(context)

paraphrase = []

for i in sentence_list:
    a = get_response(i,1)
    paraphrase.append(a)

paraphrase2 = [' '.join(x) for x in paraphrase]
paraphrase3 = [' '.join(x for x in paraphrase2) ]
paraphrased_text = str(paraphrase3).strip('[]').strip("'")

# print(paraphrased_text)

keyboard = Controller()
time.sleep(15)

for i in paraphrased_text:
    keyboard.press(i)
    keyboard.release(i)
    time.sleep(0.25)
