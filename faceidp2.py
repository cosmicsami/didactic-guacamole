import cv2,numpy,os
path="peops"
face_cascade=cv2.CascadeClassifier(r"images\haarcascade_frontalface_default.xml")
(images,labels,names,id)=([],[],{},0)
for (subdirs,dires,files)in os.walk(path):
    for subdir in dires:
        names[id]=subdir
        subjectpath=os.path.join(path,subdir)
        for filename in os.listdir(subjectpath):
            path2=subjectpath+"/"+filename
            label=id
            img=cv2.imread(path2) 
            images.append(img)
            labels.append(label)
        id=id+1
images,labels=[numpy.array(lis) for lis in [images,labels]]
model=cv2.face.LBPHFaceRecognizer_create()
model.train(images,labels)

webcam=cv2.VideoCapture(0)
while True:
    _,image=webcam.read()
    facegrey=cv2.cvtColor(image)