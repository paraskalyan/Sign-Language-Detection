
from tkinter import *
from tkinter.ttk import *
import tkinter as tk
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from PIL import Image, ImageTk
from gtts import gTTS
from io import BytesIO
import pygame

class App:
    def mainFun(self):
        self.win = tk.Tk()
        # self.win.configure(background= 'black')
        self.win.geometry("800x800")
        self.win.resizable(False, False)
        self.f1 = LabelFrame(self.win)
        self.f1.pack()
        self.L1 = Label(self.f1)
        self.L1.pack()
        # Create a Label to capture the Video frames
        # self.label =Label(self.win)
        # self.label.grid(row=0, column=0)
        self.l = Label(self.win, text = "WORD:- ")
        self.l.config(font =("Courier", 17))
        self.l.place(x=50, y= 520)
        
        # self.title = Label(self.win, text = "Sign Language Detection")
        # self.title.config(font =("Courier", 17))
        # self.title.place(x=300, y= 10)
        self.l2 = Label(self.win, text = "SENTENCE:- ")
        self.l2.config(font =("Courier", 17))
        self.l2.place(x=50, y= 570)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480) 
        self.detector = HandDetector(maxHands=1)
        self.predictLabel = Label(self.win)
        self.predictLabel.place(x=140, y= 520)
        self.sentLabel = Label(self.win)
        self.sentLabel.place(x=180, y= 570)
        self.accLabel = Label(self.win)
        self.accLabel.place(x=170, y= 520)
        self.b1 = Button(self.win, text = 'Save', command = self.printSent).place(x = 170, y = 600)
        self.b2 = Button(self.win, text = 'Speak', command = self.speak).place(x = 250, y = 600)

        self.handLabel = Label(self.win)
        self.handLabel.place(x=50,y = 100)
        self.imgSize = 300
        self.offset = 20
        self.classifier = Classifier("model-ALL/keras_model.h5", "model-ALL/labels.txt")
        self.s = ''
        self.labels = ["A", "B", "C","D","E","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
        self.key = cv2.waitKey(0)
        self.canvas = Canvas(self.win, width=100, height=100)
        self.canvas.pack()
        pygame.init()
        pygame.mixer.init() 
    def checkHand(self, img):
        hands, img2 = self.detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            print(x,y,w,h)
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]   
            aspectRatio = h / w
            self.predict(imgCrop, imgWhite, aspectRatio,h,w)
            
            

    def capture(self):
        while True:
            img = self.cap.read()[1]
            # imgOutput = img.copy()
            cv2image= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = img2)
            self.L1['image'] = imgtk
            self.win.update()
            self.checkHand(cv2image)
        self.win.mainloop()
       
    def predict(self, imgCrop,imgWhite, aR,h,w):
        if aR > 1:
            k = self.imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((self.imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, self.index = self.classifier.getPrediction(imgWhite, draw=False)
           
            
            print(prediction, self.index)

        else:
            k = self.imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((self.imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, self.index = self.classifier.getPrediction(imgWhite, draw=False)
            print(prediction,self.index) 
        self.predictLabel.config(text=self.labels[self.index], font =("Courier", 17))
        self.accLabel.config(text =str(int(prediction[self.index]*100)) +"%" , font =("Courier", 17))
        # cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        # cv2.putText(imgOutput, str(prediction[index]*100), (x+60, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
    # def remove_bg(self, imgxx):
    #     imgOut = segmentor.removeBG(imgxx, (255, 8, 255))
    #     cv2.imshow("Image Out", imgOut)


    def printSent(self):
        self.s += self.labels[self.index]
        self.sentLabel.config(text = self.s, font =("Courier", 17))

    def speak(self, language='en'):
        mp3_fo = BytesIO()
        tts = gTTS(self.s, lang=language)
        tts.write_to_fp(mp3_fo)
        pygame.mixer.music.load(mp3_fo, 'mp3')
        pygame.mixer.music.play()

obj = App()
obj.mainFun()
obj.capture()

    
