#import the necessary libraries
import cv2

#  OpenCV (cv2) provides a Haar Cascade classifier that is based on the Viola-Jones algorithm.
#  The Haar Cascade classifier is a machine learning-based approach that uses a set of features
#  and a classifier to detect objects in an image.
#  It is relatively fast and can be used to detect faces and other objects.
#  The Haar Cascade classifier is also relatively easy to use and does not require extensive training.
#  However, it may not be as accurate as YoloV8 and may struggle with detecting faces at different
#  angles or under varying lighting conditions.

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define a function to detect faces in an image using the loaded classifier
def detect_faces(img):
    # Convert the image to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces using the Haar cascade classifier with some parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Return the coordinates of the bounding boxes for the detected faces
    return faces

# Open the default video capture device (webcam)
# cap = cv2.VideoCapture(0)

# # Start an infinite loop to capture frames from the webcam and detect faces
# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect faces in the current frame using the detect_faces function
#     faces = detect_faces(frame)

#     # Draw a rectangle around each detected face in the current frame
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # Display the current frame with the detected faces
#     cv2.imshow('Faces', frame)

#     # Check if the user has pressed the 'q' key to exit the loop
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Release the video capture device and close all windows
# cap.release()
# cv2.destroyAllWindows()

img1=cv2.imread("INTERNSHIPS/CODECLAUSE/AI/img1.jpg")
img2=cv2.imread("INTERNSHIPS/CODECLAUSE/AI/img2.jpg")

faces1 = detect_faces(img1)
faces2 = detect_faces(img2)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces1:
    cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)

for (x, y, w, h) in faces2:
    cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the images with the detected faces
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
