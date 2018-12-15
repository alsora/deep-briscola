import random
import threading

from tkinter import *
import tkinter as tk
import tkinter.messagebox
from PIL import ImageTk, Image


class Card():

    def __init__(self, card_front_image, card_rear_image):
        pass

'''
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

        cards = Image.open('assets/cards.jpg')
        photo_cards = ImageTk.PhotoImage(cards)

        self.num_sprintes = 4
        self.last_img = None
        self.images = [tk.subimage(32*i, 0, 32*(i+1), 48) for i in range(self.num_sprintes)]

        background_object = self.canvas.create_image([0, 0], image=photo_background, anchor = tk.NW)
        self.img_object = self.canvas.create_image([0, 0], image=photo_rear, anchor = tk.NW)

        self.move()

        self.window.mainloop()

    def move(self):
        self.canvas.move(self.img_object, 10,0)
        self.window.after(500, self.move)
'''

class Visualizer(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        cards = Image.open('assets/cards.jpg')
        #cards = cards.resize((1000,1000), Image.ANTIALIAS)
        self.spritesheet = ImageTk.PhotoImage(cards)
        print (self.spritesheet)
        self.num_sprintes = 20
        self.last_img = None
        self.images = [self.subimage(350*i, 0, 350*(i+1), 598) for i in range(self.num_sprintes)]
        self.canvas = tk.Canvas(self, width=1500, height=1500)
        self.canvas.pack()
        self.updateimage(0)

    def subimage(self, l, t, r, b):
        print(l,t,r,b)
        dst = tk.PhotoImage()
        dst.tk.call(dst, 'copy', self.spritesheet, '-from', l, t, r, b, '-to', 0, 0)
        return dst

    def updateimage(self, sprite):
        self.canvas.delete(self.last_img)
        #where to center the image and what to draw
        self.last_img = self.canvas.create_image(250, 350, image=self.images[sprite])
        self.after(300, self.updateimage, (sprite+1) % self.num_sprintes)



if __name__ == '__main__':
    visualizer = Visualizer()
    visualizer.mainloop()