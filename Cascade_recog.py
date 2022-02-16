import cv2, numpy, os,time
size = 1
classifier = 'haarcascade_frontalface_default.xml'
image_dir = 'images'
print("Face Recognition Starting ...")

(images, labels, names, id) = ([], [], {}, 0)


for (subdirs, dirs, files) in os.walk(image_dir):

    
    for subdir in dirs:
        names[id] = subdir
        
        subjectpath = os.path.join(image_dir, subdir)

        
        for filename in os.listdir(subjectpath):

            
            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                print("Skipping "+filename+", wrong file type")
                continue
            path = subjectpath + '/' + filename
            label = id

            
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(im_width, im_height) = (120, 102)


(images, labels) = [numpy.array(lis) for lis in [images, labels]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
haar_cascade = cv2.CascadeClassifier(classifier)
webcam = cv2.VideoCapture(1)  
while True:
    
    rval = False
    while(not rval):
        
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")
    startTime = time.time()
   
    frame=cv2.flip(frame,1) 

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

   
    faces = haar_cascade.detectMultiScale(mini)
    for i in range(len(faces)):
        face_i = faces[i]

        
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        start =(x, y)
        end =(x + w, y + h)
        
        prediction = model.predict(face_resize)
        cv2.rectangle(frame,start , end, (0,255,0), 3) 
        cv2.rectangle(frame, (start[0],start[1]-20), (start[0]+120,start[1]), (0, 255, 255), -3) 
        if prediction[1]<85 : 
            cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),thickness=2)
            print('%s - %.0f' % (names[prediction[0]],prediction[1]))
        else:
            cv2.putText(frame,("Unknown {} ".format(str(int(prediction[1])))),(x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),thickness=2)
            print("Unknown -",prediction[1])
    endTime = time.time()
    fps = 1/(endTime-startTime)   
    cv2.rectangle(frame,(30,48),(130,70),(0,0,0),-1)
    cv2.putText(frame,"Fps : {} ".format(str(int(fps))),(34,65),cv2.FONT_HERSHEY_SIMPLEX,0.6,(22,22,156),2)
    cv2.imshow('Recognition System', frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
                break
webcam.release()
cv2.destroyAllWindows()                
