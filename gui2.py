from tkinter import*
from tkinter import filedialog
from tkinter.ttk import Progressbar
from PIL import ImageTk, Image, ImageEnhance
import cv2
import time
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import model_from_json
import os

global n,next1 
n = 2

#loading CNN Files#
#1#
json_file = open('cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#2#
#second cnn#
json_file = open('cnn4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model2 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model2.load_weights("model4.h5")
print("Loaded model from disk")

#backend Functions

def firstcnn():
    n=0
    path = "data3"
    images1=[]

    for i in os.listdir(path):
    
        test_image = image.load_img(str(path)+"/"+i, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
    
        if result[0][0] == 1:
            prediction = 'BH'
            images1.append(i)
        else:
            prediction = 'FH'

    return images1


def selectivesearch2(arr1):
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    image_dir = []
    n=0
    path1 = "resu"
    path = "data3"
    


    for i in arr1:
        im = cv2.imread(os.path.join(path,i))
        ss.setBaseImage(im)
        ss.switchToSelectiveSearchFast()
        pops = ss.process()
    #region select#
    for j in pops:
        x,y,w,h = j
        if x<1200:
            n = str(n)
            
            crp = im[x:x+w, y:y+h]
            resized = cv2.resize(crp, (224,224), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path1,n+".png"),resized)
            n = int(n)
            n+=1
    
    
def secondcnn():
    images2 = []
    path1 = "resu"
    for i in os.listdir(path1):
        test_image = image.load_img(str(path1)+'/'+i, target_size=(64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = loaded_model2.predict(test_image)
    
        if result[0][0] == 1:
            prediction = "nail"
            
            images2.append(i)
        else:
            prediction = "other"
    return images2


def output(arr2):
    print(arr2)
    n = 0
    dire="results"
    path1 = "resu"
    for i in arr2:
        
    
        n = str(n)
        img1 = cv2.imread(path1+'/'+i)
    
        cv2.imwrite(os.path.join(dire,n+".jpg"),img1)
        cv2.waitKey(0)
        n=int(n)
        n+=1


































#all frontend Functions

def selc():
    global path
    folder_selected = filedialog.askdirectory()
    k = 0
    print(folder_selected)
    for i in os.listdir(folder_selected):
        k = str(k)
        h = cv2.imread(folder_selected+"/"+i)
        cv2.imwrite(os.path.join("data3",k+".jpg"),h)
        k = int(k)
        k+=1

def openf():
    global path,next1
    next1 = Button(rightframe, text = "Next", command = next3)
    next1.pack(side=TOP)
    path = "data3/1.jpg"
    load = cv2.imread(path)
    load = cv2.resize(load, (780, 540))
    load = Image.fromarray(load)
    img = ImageTk.PhotoImage(load)
    global panel
    panel = Label(rightframe, image = img)
    panel.img = img
    panel.pack()

def next3():
    global n,path 
    n = str(n)
    path = "data3/"+n+".jpg"
    load = cv2.imread(path)
    load = cv2.resize(load, (780,540))
    load = Image.fromarray(load)

    img1 = ImageTk.PhotoImage(load)
    panel.configure(image = img1)
    panel.img = img1
    panel.update()
    n = int(n)
    n+=1

def eng():
    global path,n
    m= n-1
    m = str(m)
    path = "data3/"+m+".jpg"
    load = cv2.imread(path)
    load = cv2.resize(load, (780,540))
    load = Image.fromarray(load)
    load = ImageEnhance.Brightness(load)
    load = load.enhance(1.5)
    img1 = ImageTk.PhotoImage(load)
    panel.configure(image = img1)
    panel.img = img1
    panel.update()

def enh():
    global path,n
    m= n-1
    m = str(m)
    path = "data3/"+m+".jpg"
    load = cv2.imread(path)
    load = cv2.resize(load, (780,540))
    load = Image.fromarray(load)
    load = ImageEnhance.Brightness(load)
    load = load.enhance(0.5)
    img1 = ImageTk.PhotoImage(load)
    panel.configure(image = img1)
    panel.img = img1
    panel.update()
    


def deno():
    global path,n
    m = n-1
    m = str(m)
    path = "data3/"+m+".jpg"
    img = cv2.imread(path)
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    load = cv2.resize(dst, (780,540))
    load = Image.fromarray(load)
    img1 = ImageTk.PhotoImage(load)
    panel.configure(image = img1)
    panel.img = img1
    panel.update()

def hist():
    global path,n
    m = n-1
    m = str(m)
    path = "data3/"+m+".jpg"
    img = cv2.imread(path,0)
    cv2.imshow("prac", img)
    equ = cv2.equalizeHist(img)
    #stacking images side-by-side
    
    load = cv2.resize(equ, (780,540))
    load = Image.fromarray(load)
    img1 = ImageTk.PhotoImage(load)
    panel.configure(image = img1)
    panel.img = img1
    panel.update()

def submit():
    global next1,panel
    next1.destroy()
    panel.destroy()
    progress = Progressbar(rightframe, orient = HORIZONTAL,length = 800, mode = 'determinate')
    progress.pack(side = TOP, padx = 100)

    images12 = firstcnn()
    
    progress['value'] = 20
    rightframe.update_idletasks()
    time.sleep(1)
    
    selectivesearch2(images12)
    
    progress['value'] = 40
    rightframe.update_idletasks()
    time.sleep(1)
    
    images13 = secondcnn()
    
    
    progress['value'] = 50
    rightframe.update_idletasks()
    time.sleep(1)
    
    
    
    progress['value'] = 60
    rightframe.update_idletasks()
    time.sleep(1)
   
    progress['value'] = 80
    rightframe.update_idletasks()
    time.sleep(1)
    progress['value'] = 100
    
    output(images13)

def show():
    newwin = Toplevel(root)
    newwin.geometry("1200x800")
    path = "results"
    n = 0
    b = 0

    m=0

    for i in os.listdir(path):
        load = cv2.imread(path+"/"+i)
        load = cv2.resize(load,(80,80))
        load = Image.fromarray(load)
        img = ImageTk.PhotoImage(load)
        
        panel = Label(newwin, image = img)
        panel.img = img
        if m == 19:
            
            m = 0
            n = 0
            b += 80
    
        panel.place(x = 0+n, y = 0+b)
        n+=80
        m+=1




root = Tk()
root.geometry("1200x600+0+0")
root.configure(bg = 'Deepskyblue')
#frames
titleframe = Frame(root, bg="white")
titleframe.pack(side = TOP)
leftframe = Frame(root,bg="Deepskyblue")
leftframe.pack(side = LEFT)
rightframe = Frame(root, bg="Deepskyblue")
rightframe.pack(side = RIGHT)
lowerframe = Frame(root, bg = "deepskyblue")
lowerframe.pack(side = BOTTOM)

#buttons
sel = Button(leftframe, text = "Select Folder", command = selc)
sel.grid(row=0,column = 1,pady = 10)
opend = Button(leftframe, text = "Open Dataset", command = openf)
opend.grid(row=1,column=1,pady = 10)
Enhancement1 = Button(leftframe, text = "Enhancement1", command = eng)
Enhancement1.grid(row=2,column=1,pady = 10)
Enhancement2 = Button(leftframe, text = "Enhancement2", command = enh)
Enhancement2.grid(row=3,column=1,pady = 10)
Enhancement3 = Button(leftframe, text = "Enhancement3", command = deno)
Enhancement3.grid(row=4,column=1,pady = 10)
Enhancement4 = Button(leftframe, text ="Enhancement4", command = hist)
Enhancement4.grid(row=5, column=1, pady=10)
submit1 = Button(lowerframe, text = "Submit", padx = 10, pady = 10, command = submit)
submit1.pack()
displ = Button(lowerframe, text = "Show Results", padx = 10, pady = 10, command = show)
displ.pack()


title = Label(titleframe, text = "Image Enhancement Technology", font=("arial italic", 28), borderwidth=3, relief="ridge", padx=350, pady= 10, bg="white",fg ="#7f7fff")
title.pack()



root.mainloop()