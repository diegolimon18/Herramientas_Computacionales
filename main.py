import cv2
import os
import imutils

#Videostreaming input
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Inserted image for the video
image = cv2.imread('lentes.png', cv2.IMREAD_UNCHANGED)

#Face Detector
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width= 640)

    #Face detection
    faces = face.detectMultiScale(frame, 1.3, 5)

    for(x, y, w, h) in faces:
        #cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

        #Resize de input
        resized_image = imutils.resize(image, width=w)
        rows_image = resized_image.shape[0]
        col_image = w

        #Determining a portion of the height of the input image
        height_portion = rows_image

        dif = 0

        #If there is enough space in the face for inserting the image will appear
        if y + height_portion - rows_image >= 0:
            n_frame = frame[y + height_portion - rows_image : y + height_portion,
            x : x + col_image]
        else:
            dif = abs(y + height_portion - rows_image)
            n_frame = frame[0 : y + height_portion,
            x : x + col_image]

        #Determine the mask for the input image resized
        mask = resized_image[:, :, 3]
        mask_inv = cv2.bitwise_not(mask)

        #Create an image with black background and the hat frame and vice versa
        bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
        bg_black = bg_black[dif:, :, 0:3]
        bg_frame = cv2.bitwise_and(n_frame, n_frame, mask=mask_inv[dif:,:])

        #Add the two obtained images
        result = cv2.add(bg_black, bg_frame)
        if y + height_portion - rows_image >= 0:
            frame[y + height_portion - rows_image : y + height_portion, x : x + col_image] = result
        else:
            frame[0 : y + height_portion, x : x + col_image] = result

    cv2.imshow('frame',frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()