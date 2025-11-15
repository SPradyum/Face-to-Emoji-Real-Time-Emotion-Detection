import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# Load Keras model
emotion_model = tf.keras.models.load_model(
    r"D:\Documents\Projects\Python\Face to Emoji\model.h5"
)

emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

emoji_folder = r"D:\Documents\Projects\Python\Face to Emoji\emojis"

# -----------------------
# 1️⃣ CREATE ROOT FIRST
# -----------------------
root = tk.Tk()
root.title("Photo to Emoji")
root.geometry("1400x900")
root.configure(bg="black")

heading = Label(root, text="Photo to Emoji", fg="white", bg="black",
                font=("Arial", 40, "bold"))
heading.pack()

video_label = Label(root, bg="black")
video_label.place(x=50, y=150)

emoji_label = Label(root, bg="black")
emoji_label.place(x=900, y=350)

emotion_text = Label(root, fg="white", bg="black", font=("Arial", 30))
emotion_text.place(x=900, y=250)

# -----------------------
# 2️⃣ LOAD EMOJIS AFTER ROOT IS CREATED
# -----------------------
emoji_images = {}

for key, name in emotion_dict.items():
    path = os.path.join(emoji_folder, name.lower() + ".png")
    img = cv2.imread(path)

    if img is None:
        print("❌ Missing emoji:", path)
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    emoji_images[key] = ImageTk.PhotoImage(Image.fromarray(img))

# -----------------------
# 3️⃣ OPEN CAMERA ONCE
# -----------------------
cap = cv2.VideoCapture(0)

def update_video():
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.resize(frame, (600, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        resized = cv2.resize(roi, (48, 48)).reshape(1, 48, 48, 1)

        # Predict
        prediction = emotion_model.predict(resized, verbose=0)
        emotion_id = int(np.argmax(prediction))

        # Update Text & Emoji
        emotion_text.config(text=emotion_dict[emotion_id])
        emoji_label.config(image=emoji_images[emotion_id])

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    root.after(100, update_video)  # Smooth and low CPU

# Quit button
button = Button(root, text="Quit", command=root.destroy,
                font=("Arial", 24), fg="red")
button.pack(side=BOTTOM)

update_video()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
