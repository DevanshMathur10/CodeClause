# import necessary modules
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import pygame

# create the main window for the music player
root = tk.Tk()
root.title("Music Player")
root.geometry("450x300")

# load an image to use as a background for the music player-
myimg1 = Image.open("INTERNSHIPS/CODECLAUSE/PY/tape.jpg")
resize1 = myimg1.resize((450, 300), Image.ANTIALIAS)
bg = ImageTk.PhotoImage(resize1) 
imglbl = tk.Label(root, image=bg)
imglbl.place(x=0, y=0, relwidth=1, relheight=1)

# create a label for the music player title
name = tk.Label(root, text="MUSIC PLAYER", background='#E0DAD8', font=('Book Antiqua', 20))
name.place(x=125, y=40)

# define a global variable for the selected file
filename = None

# define a function to load a file using a file dialog
def load_file():
    global filename
    filename = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3;*.wav")])
    if filename:
        pygame.init()
        pygame.mixer.music.load(filename)
        messagebox.showinfo("Info", "Song loaded successfully")
    return filename

# define a function to play the loaded song
def play_music():
    global filename
    if filename:
        pygame.mixer.music.play()
    else:
        messagebox.showerror("Error", "Please load a song first")

# define a function to stop the currently playing song
def stop_music():
    pygame.mixer.music.stop()

# create buttons to stop, play, and load a song
stop_button = tk.Button(root, text="Stop", command=stop_music)
stop_button.place(x=125, y=220)

play_button = tk.Button(root, text="Play", command=play_music)
play_button.place(x=205, y=220)

load_button = tk.Button(root, text="Load", command=load_file)
load_button.place(x=285, y=220)

# start the main loop of the music player interface
root.mainloop()
