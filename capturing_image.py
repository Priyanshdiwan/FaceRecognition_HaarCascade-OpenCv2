
import cv2, sys, numpy, os
size = 2
classifier = 'haarcascade_frontalface_default.xml'
image_dir = 'images'#write the name of your directory
try:
    name_class = "PRIYANSH" # Enter the name of the person for who you want to create a database
except:
    print("You must provide a name")
    sys.exit(0)
path = os.path.join(image_dir, name_class)
if not os.path.isdir(path):
    os.mkdir(path)
(im_width, im_height) = (112, 92)
haar_cascade = cv2.CascadeClassifier(classifier)
webcam = cv2.VideoCapture(1)


pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
     if n[0]!='.' ]+[0])[-1] + 1


print("\n\033[94mThe program will save 40 samples. \
Make sure you cover every angle!\033[0m\n")


count = 0
pause = 0
count_max = 40   
while count < count_max:

    
    rval = False
    while(not rval):
        
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")

   
    height, width, channels = frame.shape # 640 , 480 ,3

    
    frame = cv2.flip(frame, 1)

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    
    faces = haar_cascade.detectMultiScale(mini)

    
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]

        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, name_class, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
            1,(0, 255, 255),2)

        
        if(w * 6 < width or h * 6 < height):
            print("Face too small")
        else:

            
            if(pause == 0):

                print("Saving training sample "+str(count+1)+"/"+str(count_max))

                # Save image file
                cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

                pin += 1
                count += 1

                pause = 1

    if(pause > 0):
        pause = (pause + 1) % 5
    cv2.imshow('Sampling', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
