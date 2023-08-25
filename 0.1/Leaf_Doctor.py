from  tkinter import * 
from tkinter import filedialog
from PIL import ImageTk,Image
window=Tk()
window.geometry("746x500")
label1=Label(window,text="                                                      ").grid(row=0,column=0)
label2=Label(window,text="welcome to your solution world ",font=('italic',30,'bold'),bg="cyan",padx=70,pady=20).grid(row=1,column=0)
label3=Label(window,text="                                                       ").grid(row=2,column=0)
def open_file():
      top=Toplevel()
      top.title("image  window")
      top.geometry("746x500")
      global my_image
      global file_open
      file_open =filedialog.askopenfilename(initialdir="F:\JetBrains\goeduhub training\PROJECTDATA 1",title="select a file")
     my_label=Label(top,text=file_open).grid(row=0,column=0)
     my_image=ImageTk.PhotoImage(Image.open(file_open))
     my_image_label=Label(top,image=my_image).grid(row=1,column=0)
     # button inside the second window
     button_exit=Button(top,text="Exit",padx=45,pady=20,command=top.destroy,font=('italic',20,'bold')).grid(row=3,column=0)
     
      button_prediction=Button(top,text="predict",padx=45,pady=20,font=('italic',20,'bold'),command=predict_img).grid(row=2,column=0)

# button inside the first window
button1=Button(window,text="click me",padx=30,pady=20,font=('italic',20,'bold'),command=open_file).grid(row=3,column=0)

label5=Label(window,text="                                                      ").grid(row=4,column=0)
button_quit=Button(text="Exit",padx=45,pady=20,command=window.destroy,font=('italic',20,'bold')).grid(row=5,column=0)

#predict output function
def predict_img():
      top1=Toplevel()
      top1.title("prediction window")
      top1.geometry("746x500")
      global test_img
      my_image_label=Label(top1,image=my_image).grid(row=0,column=0)
      from tensorflow.keras.models import load_model
      Detection=load_model('Plant_Disease_Detection.h5')
      from tensorflow.keras.preprocessing import image
      import matplotlib.pyplot as plt
      import numpy as np
      import cv2
      test_img=image.load_img(file_open,target_size=(48,48))
      test_img=image.img_to_array(test_img)
      test_img=np.expand_dims(test_img,axis=0)
      result=Detection.predict(test_img)
      a=result.argmax()
      classes=train_generator.class_indices
      category=[]
      for i in classes:
           category.append(i)
      for i in range(len(classes)):
            if(i==a):
                 output=category[i]
      output=Label(top1,text=output).grid(row=1,column=0)

window.mainloop()
