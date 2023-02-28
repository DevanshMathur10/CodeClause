import tensorflow as tf
from keras import layers
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Load the FER2013 dataset
data = pd.read_csv('D:/archive/fer2013.csv')

# Split the dataset into training, validation, and test sets
train_data = data[data['Usage'] == 'Training']
val_data = data[data['Usage'] == 'PublicTest']
test_data = data[data['Usage'] == 'PrivateTest']

# Extract the pixel values and emotion labels from the dataset
x_train = np.array(list(map(str.split, train_data['pixels']))).astype(np.float)
y_train = tf.keras.utils.to_categorical(train_data['emotion'], num_classes=7)
x_val = np.array(list(map(str.split, val_data['pixels']))).astype(np.float)
y_val = tf.keras.utils.to_categorical(val_data['emotion'], num_classes=7)
x_test = np.array(list(map(str.split, test_data['pixels']))).astype(np.float)
y_test = tf.keras.utils.to_categorical(test_data['emotion'], num_classes=7)

# Normalize the pixel values
x_train /= 255.0
x_val /= 255.0
x_test /= 255.0

# Reshape the pixel values into images
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_val = x_val.reshape(x_val.shape[0], 48, 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

# # Train the model
# history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_val, y_val))

# test_loss, test_acc = model.evaluate(x_test, y_test)
# print('Test accuracy:', test_acc)

# # Save the model
# model.save('INTERNSHIPS/CODECLAUSE/AI/facial_emotion_detection.h5')

################################################################################################
#USING THE TRAINED MODEL
################################################################################################

# Load the pre-trained model
model = tf.keras.models.load_model('INTERNSHIPS/CODECLAUSE/AI/facial_emotion_detection.h5')

# Define the emotions labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Define a function for detecting faces and emotions
# def detect_faces_emotions(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     for (x, y, w, h) in faces:
#         face_img = gray[y:y+h, x:x+w]
#         face_img = cv2.resize(face_img, (48, 48))
#         face_img = face_img.astype('float') / 255.0
#         face_img = np.expand_dims(face_img, axis=0)
#         face_img = np.expand_dims(face_img, axis=-1)
#         emotion_probs = model.predict(face_img)[0]
#         emotion_label = emotion_labels[np.argmax(emotion_probs)]
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(img, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     return img

# # Open the webcam and start the detection
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = detect_faces_emotions(frame)
#     cv2.imshow('Facial Emotion Detection', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Release the webcam and close the window
# cap.release()
# cv2.destroyAllWindows()

sample_indices = np.random.choice(x_test.shape[0], 16, replace=False)

# Create a figure with 4x4 subplots to display the sample images and their predicted labels
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    # Display the image
    ax.imshow(x_test[sample_indices[i]], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    # Add the predicted and true labels to the plot
    predicted_label = emotion_labels[np.argmax(model.predict(x_test[sample_indices[i]].reshape(1, 48, 48, 1)))]
    #true_label = emotion_labels[y_test[sample_indices[i]]]
    ax.set_xlabel(f'Predicted: {predicted_label}\nTrue:', fontsize=12)
plt.tight_layout()
plt.show()