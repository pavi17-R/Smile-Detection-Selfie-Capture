import cv2
import os
import time

# Base directory for Haar cascades
base_dir = r"C:\Users\pavit\Downloads\smile-selfie-capture-project\dataset"

# Load cascades
faceCascade = cv2.CascadeClassifier(os.path.join(base_dir, "haarcascade_frontalface_default.xml"))
smileCascade = cv2.CascadeClassifier(os.path.join(base_dir, "haarcascade_smile.xml"))

# Webcam setup
video = cv2.VideoCapture(0)

cnt = 1  # Counter for saved images
while True:
    success, img = video.read()
    if not success:
        print("Failed to capture video frame.")
        break

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)

    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        roi_gray = grayImg[y:y + h, x:x + w]
        smiles = smileCascade.detectMultiScale(roi_gray, 1.8, 15)

        for sx, sy, sw, sh in smiles:
            img = cv2.rectangle(img, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 3)

            # Countdown before saving the photo
            for i in range(3, 0, -1):
                cv2.putText(img, f"Capturing in {i}...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow('Live Video', img)
                cv2.waitKey(1000)  # Wait for 1 second

            # Save the photo
            print(f"Image {cnt} Saved")
            output_dir = r"C:\Users\pavit\Desktop\SmileCapture\images"
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"img{cnt}.jpg")
            cv2.imwrite(path, img)
            cnt += 1
            if cnt >= 2:
                break

    cv2.imshow('Live Video', img)

    # Exit condition: Press 'q' to quit
    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed == ord('q'):
        print("Exiting...")
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
