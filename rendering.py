import random
import threading

from tkinter import *
import tkinter as tk
import tkinter.messagebox
from PIL import ImageTk, Image


class Card():

    def __init__(self, card_front_image, card_rear_image):
        pass


class Visualizer(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.start()

    def callback(self):
        self.root.quit()

    def create_deck(self, deck):

        self.deck = []

        rear_path = 'assets/rear.png'
        rear = Image.open(rear_path)
        rear = rear.resize((125, 225), Image.ANTIALIAS)
        photo_rear = ImageTk.PhotoImage(rear)

        for card in deck:
            name = card.name.lower()
            name = name.replace(' ', '_')
            front_path = 'assets/' + name + '.png'

            front = Image.open(front_path)
            front = front.resize((125, 225), Image.ANTIALIAS)
            photo_front = ImageTk.PhotoImage(front)


    def run(self):

        self.window = tk.Tk()
        self.window.geometry("1000x1000")
        self.window.title("deep-briscola")

        self.canvas = tk.Canvas(self.window, height=1000, width=1000)
        self.canvas.grid(row=0, column=0, sticky='w')


        rear = Image.open('assets/rear.png')
        rear = rear.resize((125, 225), Image.ANTIALIAS)
        photo_rear = ImageTk.PhotoImage(rear)

        background = Image.open('assets/table.jpg')
        background = background.resize((1000,1000), Image.ANTIALIAS)
        photo_background = ImageTk.PhotoImage(background)

        background_object = self.canvas.create_image([0, 0], image=photo_background, anchor = tk.NW)
        self.img_object = self.canvas.create_image([0, 0], image=photo_rear, anchor = tk.NW)

        self.move()

        self.window.mainloop()

    def move(self):
        self.canvas.move(self.img_object, 10,0)
        self.window.after(500, self.move)

