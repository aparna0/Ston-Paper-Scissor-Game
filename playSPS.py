import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
from keras.models import model_from_json
import operator
import sys, os
from PIL import ImageTk
from random import randint
sys_img = {1:"\paper",2:"\scissors",3:"\stone"}

class App:
    def __init__(self, window, window_title, video_source=0):
         # Loading the model
         json_file = open("model-bw.json", "r")
         model_json = json_file.read()
         json_file.close()
         global loaded_model
         loaded_model = model_from_json(model_json)
         # load weights into new model
         loaded_model.load_weights("model-bw.h5")
         print("Loaded model from disk")
         self.window = window
         self.window.geometry('800x600')
         self.window.title(window_title)
         self.video_source = video_source

         #label
         frame1 = tkinter.Frame(master=window, width=850, height=50)
         global lab1
         lab1 = tkinter.Label(master = window, text ="",
            foreground = 'blue',        #fg = 'red'
            #background = 'white',     #bg = 'yellow'
            width = 25,
            height = 3,
            font=("times new roman", 20, "bold"))    
         lab1.pack()
         
         lab2 = tkinter.Label(master = frame1,text ="System Move",
            foreground = 'green',        #fg = 'red'
            #background = 'white',     #bg = 'yellow'
            width = 30,
            height = 3,
            font=("times new roman", 16, "bold"))    
         lab2.pack(side = "left")
         lab3 = tkinter.Label(master = frame1, text ="User Move",
            foreground = 'green',        #fg = 'red'
            #background = 'white',     #bg = 'yellow'
            width = 25,
            height = 3,
            font=("times new roman", 16, "bold"))    
         lab3.pack(side = "right")
         frame1.pack()
         
         # open video source (by default this will try to open the computer webcam)
         self.vid = MyVideoCapture(self.video_source)
 
         # Create a canvas that can fit the above video source size
         self.canvas1 = tkinter.Canvas(window, width = 350, height = 300)
         self.canvas1.pack(side = "left")
         self.canvas2 = tkinter.Canvas(window, width = 350, height = 300)
         self.canvas2.pack(side = "right")
 
         # Button that lets the user take a snapshot
         self.btn_snapshot=tkinter.Button(window, text="check", width=100,bg = "black",fg="white",font=("times new roman", 16, "bold"), command=self.snapshot)
         self.btn_snapshot.pack(side = "bottom")
 
         # After it is called once, the update method will be automatically called every delay milliseconds
         self.delay = 15
         self.update()
 
         self.window.mainloop()
 
    def snapshot(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
        
         if ret:
             
             cv2.destroyWindow("system image")
             x1 = int(0.5*frame.shape[1])
             y1 = 10
             x2 = frame.shape[1]-10
             y2 = int(0.5*frame.shape[1])
             # Drawing the ROI
             # The increment/decrement by 1 is to compensate for the bounding box
             cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
             # Extracting the ROI
             roi = frame[y1:y2, x1:x2]
    
             # Resizing the ROI so it can be fed to the model for prediction
             roi = cv2.resize(roi, (64, 64)) 
             roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
             _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
             cv2.imshow("gray",test_image)
             result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
             prediction = {'none': result[0][0], 
                  'paper': result[0][1], 
                  'scissor': result[0][2],
                  'stone' : result[0][3]}
             # Sorting based on top prediction
             prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

             # Displaying the predictions
             cv2.putText(frame, prediction[0][0], (350, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
             print(prediction[0][0])
             dirname = 'F:\machine learning\ML\sps\images'
             #dirname += "\"
             name = randint(1,3)
             dirname = dirname + sys_img[name] + '.png'
             #print(dirname)
             
             self.photo = ImageTk.PhotoImage(file = dirname)
             self.canvas1.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
             self.canvas1.pack(side = "left")
             #read image form given image ( 1: to specify as RGB image) 
             img = cv2.imread(dirname,1) 
             img = cv2.resize(img,(300,280))
             
             #to display read image
             cv2.imshow("system image",img)
             name = sys_img[name]
             name = name[1:]
             print("user move :",prediction[0][0])
             print("system move :",name)
             if prediction[0][0] == 'stone' :
                 if name == 'paper' :
                     #lab1.config(text = "You lose...")
                     text = "You lose..."
                     lab1.config(fg = "red")
                 elif name == 'scissors' :
                     #lab1.config(text = "You win!")
                     text = "You win!"
                     lab1.config(fg = "blue")
                 else :
                     #lab1.config(text = "Tie!")
                     text = "Tie!"
                     lab1.config(fg = "blue")
             elif prediction[0][0] == 'paper' :
                 if name == 'scissors' :
                     #lab1.config(text = "You lose...")
                     text = "You lose..."
                     lab1.config(fg = "red")
                 elif name == 'stone' :
                     #lab1.config(text = "You win!")
                     text = "You win!"
                     lab1.config(fg = "blue")
                 else :
                     #lab1.config(text = "Tie!")
                     text = "Tie!"
                     lab1.config(fg = "blue")
             elif prediction[0][0] == 'scissor' :
                 if name == 'stone' :
                     #lab1.config(text = "You lose...")
                     text = "You lose..."
                     lab1.config(fg = "red")
                 elif name == 'paper' :
                     #lab1.config(text = "You win!")
                     text = "You win!"
                     lab1.config(fg = "blue")
                 else :
                     #lab1.config(text = "Tie!")
                     text = "Tie!"
                     lab1.config(fg = "blue")
             else :
                 #lab1.config(text = "Hold your hand in box")
                 text = "Hold your hand in box"
                 lab1.config(fg = "blue")
        
             print(text)
             lab1.config(text = text)
             cv2.imshow("system image",img)
             cv2.waitKey(1000)
             cv2.destroyWindow("system image")
    
 
    def update(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
 
         if ret:
             
             x1 = int(0.5*frame.shape[1])
             y1 = 10
             x2 = frame.shape[1]-10
             y2 = int(0.5*frame.shape[1])
             roi = frame[y1:y2, x1:x2]
             roi = cv2.resize(roi, (64, 64)) 
             roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
             _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
             cv2.imshow("gray",test_image)
             result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
             prediction = {'none': result[0][0], 
                  'paper': result[0][1], 
                  'scissor': result[0][2],
                  'stone' : result[0][3]}
             # Sorting based on top prediction
             prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

             # Displaying the predictions
             cv2.putText(frame, prediction[0][0], (350, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
             #print(prediction[0][0])
             frame = frame[y1:y2, x1:x2]
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
             self.canvas2.create_image(20, 20, image = self.photo, anchor = tkinter.NW)
        
         self.window.after(self.delay, self.update)
 
 
class MyVideoCapture:
     def __init__(self, video_source=0):
         # Open the video source
         self.vid = cv2.VideoCapture(video_source)
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)
 
         # Get video source width and height
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
     def get_frame(self):
         if self.vid.isOpened():
             ret, frame = self.vid.read()
             frame = cv2.flip(frame, 1)
             
             #cv2.
             if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             else:
                 return (ret, None)
         else:
             return (ret, None)
 
     # Release the video source when the object is destroyed
     def __del__(self):
         if self.vid.isOpened():
             self.vid.release()
 
# Create a window and pass it to the Application object
App(tkinter.Tk(), "STONE-PAPER-SCISSOR GAME...")
