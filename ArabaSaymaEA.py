import cv2
import numpy as np

class Kordinat:
    def __init__(self,x,y):
        self.x=x
        self.y=y

class Sensor:
    def __init__(self,kordinat1,kordinat2,frame_weight,frame_lenght):
        self.kordinat1=kordinat1
        self.kordinat2=kordinat2
        self.frame_weight=frame_weight
        self.frame_lenght =frame_lenght
        self.mask=np.zeros((frame_weight,frame_lenght,1),np.uint8)*abs(self.kordinat2.y-self.kordinat1.y)
        self.full_mask_area=abs(self.kordinat2.x-self.kordinat1.x)
        cv2.rectangle(self.mask,(self.kordinat1.x,self.kordinat1.y),(self.kordinat2.x,self.kordinat2.y),(255),thickness=cv2.FILLED)
        self.stuation=False
        self.car_number_detected=0


Sensor1 = Sensor(Kordinat(1, 425), Kordinat(1080, 430), 500, 1080)
video=cv2.VideoCapture("video2.mp4")
fgbg=cv2.createBackgroundSubtractorMOG2()
kernel=np.ones((5,5),np.uint8)
font=cv2.FONT_HERSHEY_TRIPLEX
while True:
    ret,frame=video.read()
    # çerçeveyi yeniden boyutlandır
    cut_image=frame[100:600,100:1180]
    # çerçeve için morfoloji yapmak
    deleted_background=fgbg.apply(cut_image)
    opening_image=cv2.morphologyEx(deleted_background,cv2.MORPH_OPEN,kernel)
    ret,opening_image=cv2.threshold(opening_image,125,255,cv2.THRESH_BINARY)

    # hareketi algılama kodu
    cnts,_ = cv2.findContours(opening_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    result=cut_image.copy()

    zeros_image=np.zeros((cut_image.shape[0],cut_image.shape[1],1),np.uint8)

    # döngü ile hareket eden herhangi bir şeyi algılamak
    for cnt in cnts:
        x,y,w,h=cv2.boundingRect(cnt)
        if (w>100 and h>100 ):
            cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),thickness=2)
            cv2.rectangle(zeros_image,(x,y),(x+w,y+h),(255),thickness=cv2.FILLED)

    # arabayı tespit etmek
    mask1=np.zeros((zeros_image.shape[0],zeros_image.shape[1],1),np.uint8)

    mask_result=cv2.bitwise_and(zeros_image,zeros_image,mask=Sensor1.mask)
    white_cell_number=np.sum(mask_result==255)

    # arabayı kırmızı çizgiden geçtiğini kontrol et
    sensor_rate=white_cell_number/Sensor1.full_mask_area
    if sensor_rate>0:
        print(sensor_rate)

    # araba kırmızıdan geçiyorsa çizgi = sarı renk
    if (sensor_rate >= 1.8 and sensor_rate<2.9 and Sensor1.stuation == False):
        # kırmızı çizgiyi çiz
        cv2.rectangle(result, (Sensor1.kordinat1.x, Sensor1.kordinat1.y), (Sensor1.kordinat2.x, Sensor1.kordinat2.y),
                      (0, 0, 255), thickness=cv2.FILLED)
        Sensor1.stuation = False
        Sensor1.car_number_detected += 2
    if (sensor_rate>=0.6 and  sensor_rate<1.8 and Sensor1.stuation==False):
        # kırmızı çizgiyi çiz
        cv2.rectangle(result, (Sensor1.kordinat1.x, Sensor1.kordinat1.y), (Sensor1.kordinat2.x, Sensor1.kordinat2.y),
                      (0,255, 0,), thickness=cv2.FILLED)
        Sensor1.stuation = True
    elif (sensor_rate<0.6 and Sensor1.stuation==True) :
        # kırmızı çizgiyi çiz
        cv2.rectangle(result, (Sensor1.kordinat1.x, Sensor1.kordinat1.y), (Sensor1.kordinat2.x, Sensor1.kordinat2.y),
                      (0, 0,255), thickness=cv2.FILLED)
        Sensor1.stuation = False
        Sensor1.car_number_detected+=1
    else:
        # kırmızı çizgiyi çiz
        cv2.rectangle(result, (Sensor1.kordinat1.x, Sensor1.kordinat1.y), (Sensor1.kordinat2.x, Sensor1.kordinat2.y),(0, 0, 255), thickness=cv2.FILLED)

    cv2.putText(result,str(Sensor1.car_number_detected),(Sensor1.kordinat1.x,Sensor1.kordinat1.y+60),font,2,(255,255,255))


    cv2.imshow("video", result)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()