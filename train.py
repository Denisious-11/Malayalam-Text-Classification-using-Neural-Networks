#importing necessary libraries
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding,LSTM, Dense,Dropout,Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from preprocess import *



#load dataset
df = pd.read_csv("Data/train.csv")
print("\n[INFO] : Dataset Loaded\n")
print(df.head())
print(df.columns)


#checking null values
print( df.isnull().sum())

#printing valuecounts of label column
print(df['label'].value_counts(ascending=False))

#pie grpah on label column
df['label'].value_counts().plot(kind='pie')
plt.title(label="Analysing Label feature")
plt.show()

#bar graph on label column
df['label'].value_counts().plot(kind='barh')
plt.xlabel('Rows count')
plt.ylabel('Category')
plt.title(label="Analysing Label feature")
plt.show()

# Convert categorical values to numerical values
df.loc[df['label'] == 'business', 'label']=0
df.loc[df['label'] == 'entertainment', 'label']=1
df.loc[df['label'] == 'sports', 'label']=2
print(df.head())



# Seperating data and labels
data=df["headings"]
labels=df["label"]


data=data.apply(lambda x:cleantext(x))


#Train-Test Splitting
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.2)

print("\nTraining set")
print(x_train.shape)
print(y_train.shape)
print("\nTesting set")
print(x_test.shape)
print(y_test.shape)



#Calculate max length of the headings
def calc_max():
    review_length=[]
    for review in x_train:
        review_length.append(len(review)) 
    return max(review_length) 


max_length=calc_max()

print("Max_Length : ",max_length) #392#399


###################Word Embedding
#Feature extraction
token=Tokenizer(lower=False) #convert the review to tokens(words)
token.fit_on_texts(x_train)  #each word automatically indexed
x_train=token.texts_to_sequences(x_train)   #convert it into integers
x_test=token.texts_to_sequences(x_test)     #convert it into integers

print(x_train)
print(x_test)


###Padding(adding 0)/Truncating headings based on max value
x_train=pad_sequences(x_train,maxlen=max_length,padding="post",truncating="post")#post->back of sentence
x_test=pad_sequences(x_test,maxlen=max_length,padding="post",truncating="post")

total_words=len(token.word_index)+1 #add 1 because of 0 padding

print("Encoded x_train\n",x_train,"\n")
print("Encoded x_test\n",x_test,"\n")

#convert to array
x_train=np.asarray(x_train).astype(np.int)
y_train=np.asarray(y_train).astype(np.int)


#perform one hot encoding becz of multi-classification
#(number_of_samples, 3), where 3 denotes number of classes.
y_train = to_categorical(y_train, 3) # eg :  1   --->     [0 1 0]       eg : 0   --->   [1 0 0]
y_test = to_categorical(y_test, 3)




# def try_():
    # EMBED_DIM=32
    # LSTM_OUT=64
    # model.add(Embedding(total_words,EMBED_DIM,input_length=max_length))#(size of vocabulary,size of output vector,input length)
    # model.add(LSTM(LSTM_OUT))
    # model.add(Dense(256,activation="relu"))
    # model.add(Dropout(0.5))
    # model.add(Dense(3,activation="softmax"))

    
# ####################Building Model
# #Model Architecture
model=Sequential()#this model takes sequences of data

model.add(Embedding(total_words, 128))
model.add(Bidirectional(LSTM(64,  return_sequences=True)))
model.add(Bidirectional(LSTM(16)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation="softmax"))


#Compiling the model
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

#Printing model summary
print(model.summary())


#saving the tokenizer
with open("Trained_Model/tokenizer_check.pickle",'wb') as handle:
    pickle.dump(token,handle,protocol=pickle.HIGHEST_PROTOCOL)



#saving the model(checkpoint)
checkpoint=ModelCheckpoint("Trained_Model/my_model_check.h5",monitor="accuracy",save_best_only=True,verbose=1)#when training deep learning model,checkpoint is "WEIGHT OF THE MODEL"


#training 
history=model.fit(x_train,y_train,batch_size=128,epochs=5,validation_data=(x_test,y_test),callbacks=[checkpoint])



#plot accuracy and loss 
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('Results/acc_plot_check.png')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('Results/loss_plot_check.png')
plt.show()

#evaluating model on testing data 
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)


class_types=['business', 'entertainment','sports']
#create confusion matrix
cm = confusion_matrix(y_true, y_pred)

#plot confusion matrix
plt.figure(figsize=(10, 10))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=class_types, yticklabels=class_types)
ax.set_ylabel('Actual', fontsize=24)
ax.set_xlabel('Predicted', fontsize=24)
plt.savefig('Results/cm_check.png')
plt.show()


precision = precision_score(y_true, y_pred,average='micro')
# Calculate recall
recall = recall_score(y_true, y_pred,average='micro')
# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

precision_percentage = precision * 100
recall_percentage = recall * 100
accuracy_percentage = accuracy * 100

print("Precision:", precision_percentage, "%")
print("Recall:", recall_percentage, "%")
print("Accuracy:", accuracy_percentage, "%")



