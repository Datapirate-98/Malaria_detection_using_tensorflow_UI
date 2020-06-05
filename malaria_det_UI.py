

import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

from PIL import Image, ImageTk

import malaria_det_UI_support
import os.path
from tkinter import messagebox
import cv2,glob
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    root = tk.Tk()
    top = Toplevel1 (root)
    malaria_det_UI_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_Toplevel1(root, *args, **kwargs)' .'''
    global w, w_win, root
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    top = Toplevel1 (w)
    malaria_det_UI_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
    
    
    def clear(self):
            length_1 = len(self.Entry1.get())
            self.Entry1.delete(length_1 - 1, "end")
        
    def classify(self):
            filename = self.Entry1.get()
            if filename == "":
                messagebox.showinfo("Error", "Invalid Subject or Room")
            else:
                messagebox.showinfo("Done", "RESULT")
                
                model = tf.keras.models.load_model("malaria_tensor.h5")

                classifier=cv2.CascadeClassifier("cascade3.xml")

                labels_dict={0:"Malaria",1:"Not-Malaria"}
                color_dict={0:(0,0,255),1:(0,255,0)}

                img = cv2.imread("images_for_testing/"+ filename)
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                cells=classifier.detectMultiScale(gray,1.3,5)  

                for x,y,w,h in cells:
                    try:

                        cell_img=gray[y:y+w,x:x+w]
                        resized=cv2.resize(cell_img,(128, 128))
                        normalized=resized/255.0
                        reshaped=np.reshape(normalized,(1, 128, 128,1))
                        result=model.predict(reshaped)
                        print(result)
                        label=np.argmax(result,axis=1)[0]
                        print(label)

                    except Exception as e:
                        print("Exception", e)

                    cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
                    cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
                    cv2.putText(img, labels_dict[label], (x-5, y+10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)


                cv2.imshow("window", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    
    
    
    
    
    def __init__(self, top=None):   
        
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        font9 = "-family {8514oem} -size 11"

        top.geometry("832x656+650+150")
        top.minsize(148, 1)
        top.maxsize(1924, 1055)
        top.resizable(1, 1)
        top.title("New Toplevel")
        top.configure(background="#00ffff")

        self.Label1 = tk.Label(top)
        self.Label1.place(relx=0.125, rely=0.509, height=33, width=131)
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font=font9)
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''Sample Image''')

        self.Entry1 = tk.Entry(top)
        self.Entry1.place(relx=0.3, rely=0.509,height=34, relwidth=0.45)
        self.Entry1.configure(background="white")
        self.Entry1.configure(disabledforeground="#a3a3a3")
        self.Entry1.configure(font="TkFixedFont")
        self.Entry1.configure(foreground="#000000")
        self.Entry1.configure(insertbackground="black")

        self.Button1 = tk.Button(top)
        self.Button1.place(relx=0.475, rely=0.601, height=33, width=76)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Clear''')
        self.Button1.configure(command = self.clear)

        self.Button2 = tk.Button(top)
        self.Button2.place(relx=0.363, rely=0.782, height=53, width=266)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(disabledforeground="#a3a3a3")
        self.Button2.configure(font=font9)
        self.Button2.configure(foreground="#000000")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(pady="0")
        self.Button2.configure(text='''CHECK''')
        self.Button2.configure(command = self.classify)

        self.Label2 = tk.Label(top)
        self.Label2.place(relx=0.0, rely=0.0, height=326, width=832)
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(foreground="#000000")
        photo_location = os.path.join(prog_location,"label_img.jpg")
        global _img0
        _img0 = ImageTk.PhotoImage(file=photo_location)
        self.Label2.configure(image=_img0)
        self.Label2.configure(text='''Label''')

if __name__ == '__main__':
    vp_start_gui()





