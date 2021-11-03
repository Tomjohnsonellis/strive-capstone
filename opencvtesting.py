import sys
import cv2
import torch


def output_versions():
    print("-"*25, "Version Info", "-"*25)
    print("Python version: {}".format(sys.version))
    print("OpenCV version: {}".format(cv2.__version__))
    print("PyTorch version: {}".format(torch.__version__))


    print("-"*64)
    return


# Load webcam
cap = cv2.VideoCapture(0)

# Close all windows
def close_all_windows():
    cap.release()
    cv2.destroyAllWindows()
    return


# Create haar cascades
frontCascPath = "haar/haarcascade_frontalface_default.xml"
frontFaceCascade = cv2.CascadeClassifier(frontCascPath)
## Profile face detection deemed useless
# sideCascPath = "haar/haarcascade_profileface.xml"
# sideFaceCascade = cv2.CascadeClassifier(sideCascPath)

# Detect faces in frame
def detect_faces(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = frontFaceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces, len(faces)





if __name__ == "__main__":



    output_versions()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()


        # Detect faces in frame
        faces, num_faces = detect_faces(frame)
        if num_faces > 0:
            print("Found {} faces!".format(num_faces))
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)



        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    close_all_windows()