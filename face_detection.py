#Importing cv2
import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model('D:/chuyen_nganh/20221/machine_learning/project/Nhap_mom_ML-hao/Nhap_mom_ML-hao/models1')
test_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
print('OK')

#Loading cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, frame): 
    # We create a function that takes as input the image in black and white (gray) 
    #and the original image (frame), and that will return the same image with the detector rectangles. 
    
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    #scaleFactor--specifying how much the image size is reduced at each image scale
    #minNeighbors--specifying how many neighbors each candidate rectangle should have
    
    for (x, y, w, h) in faces: # For each detected face: (faces is the tuple of x,y--point of upper left corner,w-width,h-height)
        cv2.rectangle(frame, (x,y-30), (x+w,y+h+35), (255,0,0), 3)  #frame on which rectangle will be there,top-left,bottom-right,color,thickness
        img_cat = frame[x:y-30, x+w:y+h+25]
        img_age = np.resize(img_cat, (3, 120, 120, 3))
        img_age = img_age.astype('float32')
        print('img')
        print(len(img_age))
        print('frame')
        print(len(frame))
        # img_age/=255
        print('okk')
        # model.predict(img_age)
        img_pedict = test_generator.flow(img_age, batch_size=32, shuffle=True)
        print(h)

        output_predict = np.squeeze(model.predict(img_pedict)).item(0)
        print('pre')
        print(output_predict)
        col = (0, 255, 0)
        cv2.putText(frame, str(output_predict), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, h/200, col ,2)
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image. (range from y to y+h)
        #This region is calculated as to save computation time to again search for eyes in whole image
        #It's better to detect a face and take the region of interest i.e. face and find eyes in it
        roi_color = frame[y-30:y+h, x:x+w] # We get the region of interest in the colored image.
        
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1,minNeighbors=22)
        
        for (ex, ey, ew, eh) in eyes: # For each detected eye: (Again retrieving x,y,w,h)
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 3)
        
    return frame # We return the image with the detector rectangles.  

# def age_predict()