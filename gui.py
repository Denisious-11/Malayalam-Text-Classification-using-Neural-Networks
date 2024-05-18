from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import load_model
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from preprocess import *

#load trained model
loaded_model=load_model("Trained_Model/my_model.h5")

#load tokenizer
with open("Trained_Model/tokenizer.pickle",'rb') as handle:
    token=pickle.load(handle)

max_length=392

#window creation
a = Tk()
a.title("Malayalam Entity Recognition")
a.geometry("1000x500")
a.minsize(1000,500)
a.maxsize(1000,500)


def prediction():
    
    alltext=text1.get("1.0",'end')
    if alltext=='' or alltext=='\n':
        message.set("fill the empty field!!!")
    else:
        list_box.insert(1, "Preprocessing")
        list_box.insert(2, "")
        list_box.insert(3, "Perform word embedding")
        list_box.insert(4, "")
        list_box.insert(5, "Loading Trained Model")
        list_box.insert(6, "")
        list_box.insert(7, "Prediction")
        message.set("")

        pre1=cleantext(alltext)
        pre2=[pre1]

        tokenize_words=token.texts_to_sequences(pre2)
        tokenize_words=pad_sequences(tokenize_words,maxlen=max_length,padding="post",truncating="post")
        

        result=loaded_model.predict(tokenize_words)
        f_result=np.argmax(result)
        print(f_result)

        if f_result==0:
            print("[Result] : Business")
            output="[Result] : Business"
        if f_result==1:
            print("[Result] : Entertainment")
            output="[Result] : Entertainment"
        if f_result==2:
            print("[Result] : Sports")
            output="[Result] : Sports"

        out_label.config(text=output)



def Check():
    global f
    f.pack_forget()

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="green yellow")
    f1.place(x=0, y=0, width=760, height=250)
    f1.config()

    input_label = Label(f1, text="INPUT", font="arial 16", bg="green yellow")
    input_label.pack(padx=0, pady=10)

    
    global message
    message = StringVar()

    global text1
    text1=Text(f1,height=8,width=70)
    text1.pack()


    msg_label = Label(f1, text=
        "", textvariable=message,
                      bg='green yellow').place(x=330, y=185)

    predict_button = Button(
        f1, text="Find", command=prediction, bg="pink")
    predict_button.pack(side="bottom", pady=16)
    global f2
    f2 = Frame(f, bg="steel blue2")
    f2.place(x=0, y=250, width=760, height=500)
    f2.config(pady=20)

    result_label = Label(f2, text="RESULT", font="arial 16", bg="steel blue2")
    result_label.pack(padx=0, pady=0)

    global out_label
    out_label = Label(f2, text="", bg="steel blue2", font="arial 16")
    out_label.pack(pady=70)

    f3 = Frame(f, bg="mistyrose")
    f3.place(x=760, y=0, width=240, height=690)
    f3.config()

    name_label = Label(f3, text="Process", font="arial 14", bg="mistyrose")
    name_label.pack(pady=20)

    global list_box
    list_box = Listbox(f3, height=12, width=31)
    list_box.pack()



def Home():
    global f
    f.pack_forget()

    f = Frame(a, bg="light goldenrod")
    f.pack(side="top", fill="both", expand=True)

    # front_image = Image.open("Results/home1.jpg")
    # front_photo = ImageTk.PhotoImage(front_image.resize((a.winfo_width(), a.winfo_height()), Image.ANTIALIAS))
    # front_label = Label(f, image=front_photo)
    # front_label.image = front_photo
    # front_label.pack()

    home_label = Label(f, text="Malayalam Entity Recognition",
                       font="arial 35", bg="light goldenrod")
    home_label.place(x=220, y=200)


f = Frame(a, bg="light goldenrod")
f.pack(side="top", fill="both", expand=True)

# front_image1 = Image.open("Results/home1.jpg")
# front_photo1 = ImageTk.PhotoImage(front_image1.resize((1000,650), Image.ANTIALIAS))
# front_label1 = Label(f, image=front_photo1)
# front_label1.image = front_photo1
# front_label1.pack()

home_label = Label(f, text="Malayalam Entity Recognition",
                   font="arial 35", bg="light goldenrod")
home_label.place(x=220, y=200)

m = Menu(a)
m.add_command(label="Homepage", command=Home)
checkmenu = Menu(m)
m.add_command(label="Test", command=Check)
plotmenu=Menu(m)
a.config(menu=m)


a.mainloop()
