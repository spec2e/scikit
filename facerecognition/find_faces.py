import dlib
from skimage import io

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()


# Load the image into an array
image = io.imread("many_faces.jpg")

# Run the HOG face detector on the image data.
# The result will be the bounding boxes of the faces in our image.
detected_faces = face_detector(image, 1)

print("I found {} faces".format(len(detected_faces)))

# Open a window on the desktop showing the image
#win.set_image(image)

# Loop through each face we found in
#  the image
for i, face_rect in enumerate(detected_faces):
    crop = image[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]
    io.imsave("found_faces/img_" + str(i) + ".jpg", crop)
