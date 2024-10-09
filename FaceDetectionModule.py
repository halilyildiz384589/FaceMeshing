import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__ (self, minDetectionCon = 0.5): #self sınıfı örnek aldığımızı belirtir
        self.minDetectionCon = minDetectionCon #minDetection ı sınıf içinde tanımladık
        self.mpFaceDetection = mp.solutions.face_detection #face_detection ı içeri aktardık
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True): #bu kod ile findFaces sınıfına girip özelliklerini kullancağımızı söyledik
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #mediapipe rgb formatında bekler
        self.results = self.faceDetection.process(imgRGB) #proses metodu ile yeni RGB li görüntüyü faceDetection a gönderiyorum.
        print(self.results)

        bboxs = [] #bbox adında bir liste oluşturdum
        if self.results.detections: #eğer algılama sonucu varsa, bu bloğa gir
            for id, detection in enumerate(self.results.detections): #bloğa girmeyi sağlayan her resimi hesapla ve ID ile detection sonuçlarını tut
                bboxC = detection.location_data.relative_bounding_box #bboxC ile çizilen dikdörtgen şemayı alıp saklayacağız
                ih, iw, ic = img.shape #c kanal sayısıdır. RGB 3 kanallıdır
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * iw), \
                    int(bboxC.width * iw), int(bboxC.height * iw)  # iw = image weigth, bbox ile koordinatları hesapladık ve tuttuk
                bboxs.append([id, bbox, detection.score]) #bboxs listesinde id ve detections sonuçlarını tuttuk

                if draw: #draw True ise içeri gir
                    img = self.fancyDraw(img, bbox) #koordinatları ve resimi al img e ata. fancydraw aşağıda yazıldı
                    cv2.putText(img, f'{int(detection.score[0] * 100)} %', #algılama puanını 100 üzerinden göster
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, #(bbox[0], bbox[1] - 20) ile nerede göstereceğini belirledik
                                2, (255, 0, 255), 2)

        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=7, rt=1): #kenar köşelere kalınlık çiziyoruz.
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (255, 0, 255), rt) #dörtgenin şeklini söyledik
        #En sol nokta x, y
        cv2.line(img, (x,y), (x+l, y), (255,0,255), t)
        cv2.line(img, (x, y), (x, y+ l), (255, 0, 255), t)
        # En sol üst nokta x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        #En alt nokta x, y
        cv2.line(img, (x,y1), (x+l, y1), (255,0,255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # En sol alt nokta x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img



def main():
    cap = cv2.VideoCapture(0)

    pTime = 0

    detector = FaceDetector()

    while True:
        success, img = cap.read()

        img, bboxs = detector.findFaces(img)

        print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 2)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# main kodunu çalıştır
if __name__ == "__main__":
    main()