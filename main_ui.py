from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

def show_message_box(final_label):
    if not hasattr(show_message_box, "popup"):
        show_message_box.popup = tk.Toplevel()
        show_message_box.popup.title("Gender Detection")
        show_message_box.label = tk.Label(show_message_box.popup, text="")
        show_message_box.label.pack(padx=20, pady=20)
    show_message_box.label.config(text=f"Detected gender: {final_label}")
    show_message_box.popup.update()

def process_frame():
    global frame_count, predictions, running, webcam

    if not running:
        return

    # Read frame from webcam
    status, frame = webcam.read()

    # Apply face detection
    face, confidence = cv.detect_face(frame)

    # Loop through detected faces
    for idx, f in enumerate(face):
        # Get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on face
        conf = model.predict(face_crop)[0]  # model.predict returns a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        predictions.append(idx)
        frame_count += 1

        if frame_count >= 10:
            avg_prediction = np.mean(predictions)
            final_label = classes[int(round(avg_prediction))]

            # Show/update message box in the main thread
            root.after(0, lambda: show_message_box(final_label))

            # Reset for next set of frames
            predictions = []
            frame_count = 0

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display output
    cv2.imshow("gender detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        webcam.release()
        cv2.destroyAllWindows()
        return

    # Continue processing the next frame
    root.after(10, process_frame)

def start_detection():
    global running, webcam
    running = True
    webcam = cv2.VideoCapture(0)
    process_frame()

# Initialize global variables
frame_count = 0
predictions = []
running = False
webcam = None

# Load model
model = load_model('gender_detection.model')
classes = ['man', 'woman']

# Tkinter setup
root = tk.Tk()

# Load and display the image
image = Image.open("b.jpg")
image = image.resize((1400, 700), Image.ANTIALIAS)
tk_image = ImageTk.PhotoImage(image)

label = tk.Label(root, image=tk_image)
label.pack()

# Create and place the frame
frame = tk.Frame(root, bg="white", bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')

# Create and place the heading label
heading = tk.Label(frame, text="Real-time Gender Detection", font=('Helvetica', 18, 'bold'), bg="black", fg="white")
heading.place(relwidth=1, relheight=1)

button = tk.Button(root, text="Start Detection", font=('Helvetica', 14), bg="black", fg="white", command=start_detection)
button.place(relx=0.5, rely=0.5, anchor='center')

# Start the Tkinter event loop
root.mainloop()
