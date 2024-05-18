#Importing necessary libraries
from tensorflow.keras.models import load_model
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from preprocess import *

loaded_model=load_model("Trained_Model/my_model.h5")
# comment=str(input('Enter Comment : '))
comment="ഈ താരപുത്രി മിന്നിക്കും, സുരേഷ് ഗോപിയുടെ മകള്‍ നായികയാവുന്നു! താനിതൊന്നും അറിഞ്ഞില്ലെന്ന് ഗോകുല്‍!!"

pre1=cleantext(comment)
# print("PRE 1 : ",pre1)
# print("PRE 2 : ",pre2)

# print(type(pre2))

pre2=[pre1]

max_length=392

#load tokenizer
with open("Trained_Model/tokenizer.pickle",'rb') as handle:
    token=pickle.load(handle)

tokenize_words=token.texts_to_sequences(pre2)
tokenize_words=pad_sequences(tokenize_words,maxlen=max_length,padding="post",truncating="post")
# print("tokenized:",tokenize_words)



result=loaded_model.predict(tokenize_words)
print(result)
f_result=np.argmax(result)
print(f_result)

if f_result==0:
    print("[Result] : Business")
if f_result==1:
    print("[Result] : Entertainment")
if f_result==2:
    print("[Result] : Sports")

