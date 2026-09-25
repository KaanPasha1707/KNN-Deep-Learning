import os, numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from img2vec_pytorch import Img2Vec
import json, random
from collections import Counter 
import matplotlib.pyplot as plt

with open("C:/Users/USER/OneDrive/Belgeler/auair2019annotations/annotations.json", "r") as f:
    anno = json.load(f)

etiketHaritası = {}
for eti in anno['annotations']:
    resimAdı = eti['image_name']
    bboxlar = eti.get('bbox', [])
   
    if len(bboxlar) > 0:
        etiketHaritası[resimAdı] = bboxlar

imageKlasör = "C:/Users/USER/OneDrive/Belgeler/auair2019data (1)/images"     
tümResimler = os.listdir(imageKlasör)

nesneliResimler = []
for r in tümResimler:
    if r in etiketHaritası:
        nesneliResimler.append(r)

print(f"toplam {len(tümResimler)} resimden {len(nesneliResimler)} tanesinde nesne bulundu")        

random.seed(42)
random.shuffle(nesneliResimler)

toplamResimSayısı = len(nesneliResimler)
trainSınır = int(toplamResimSayısı * 0.70)
valSınır = int(toplamResimSayısı * 0.85)

trainResimleri = nesneliResimler[:trainSınır]
valResimleri = nesneliResimler[trainSınır:valSınır]
testResimleri = nesneliResimler[valSınır:]

print(f"Resim dağılımı -> Train {len(trainResimleri)}, Val: {len(valResimleri)}, Test: {len(testResimleri)}")

def öznitelikVeEtiketTopla(resimListesi, img2vecModel):
    vektörler = []
    etiketler = []
    count = 0

    for isim in resimListesi:
        yol = os.path.join(imageKlasör, isim)
        mevcutBboxlar = etiketHaritası.get(isim, [])

        try:
            tamResim = Image.open(yol).convert('RGB')
            for box in mevcutBboxlar:
                sınıf = box['class']
                top, left = int(box['top']), int(box['left'])
                width, height = int(box['width']), int(box['height'])

                if width <= 0 or height <= 0:
                    continue
                kırpılmışResim = tamResim.crop((left, top, left+width, top+height))
                vektör = img2vec.get_vec(kırpılmışResim)

                vektörler.append(vektör.flatten())
                etiketler.append(sınıf)
            count += 1
            if count % 200 == 0:
               print(f"ilerleme: {count}")
        except Exception as e:
            continue

    return np.array(vektörler), np.array(etiketler)

          # 'resnet18'
modelTipi = 'alexnet' 
vektörKök = f"Veri_{modelTipi}"

if os.path.exists(f"{vektörKök}_xTrain.npy"):
    print("\nÖnceden kaydedilmiş nesne vektörleri bulunuyor. Doğrudan yükleniyor...")
    xTrain = np.load(f"{vektörKök}_xTrain.npy")
    yTrain = np.load(f"{vektörKök}_yTrain.npy")
    xVal = np.load(f"{vektörKök}_xVal.npy")
    yVal = np.load(f"{vektörKök}_yVal.npy")
    xTest = np.load(f"{vektörKök}_xTest.npy")
    yTest = np.load(f"{vektörKök}_yTest.npy")
    print("Yükleme tamamlandı")

else:
    print(f"\n{modelTipi} modeli kullanılarak nesne tabanlı öznitelik çıkarılıyor...")    
    
    img2vec = Img2Vec(model=modelTipi, cuda=False)

    print("\n---Eğitim Seti Başlıyor---")
    xTrain, yTrain = öznitelikVeEtiketTopla(trainResimleri, img2vec)

    print("\n---Validation Seti Başlıyor---")
    xVal, yVal = öznitelikVeEtiketTopla(valResimleri, img2vec)

    print("\n---Test Seti Başlıyor---")
    xTest, yTest = öznitelikVeEtiketTopla(testResimleri, img2vec)

    np.save(f"{vektörKök}_xTrain.npy", xTrain)
    np.save(f"{vektörKök}_yTrain.npy", yTrain)
    np.save(f"{vektörKök}_xVal.npy", xVal)
    np.save(f"{vektörKök}_yVal.npy", yVal)
    np.save(f"{vektörKök}_xTest.npy", xTest)
    np.save(f"{vektörKök}_yTest.npy", yTest)
    print("\nTüm alt veri kümeleri başarıyla '.npy' olarak diske kaydedildi")

print(f"\nEğitim veri kümesindeki nesne sayısı: {len(xTrain)}, Dağılım: {Counter(yTrain)}")
print(f"Geçerleme veri kümesindeki nesne sayısı: {len(xVal)}, Dağılım: {Counter(yVal)}")
print(f"Test veri kümesindeki nesne sayısı: {len(xTest)}, Dağılım: {Counter(yTest)}")    

pcaBoyutları = [10,20,30]
kDeğerleri = [1,3,5]

enİyiSkor = -1
enİyiPca = None
enİyiK = None
enİyiModel = None
enİyiPcaObjesi = None

knnBaseline = KNeighborsClassifier(n_neighbors=5)
knnBaseline.fit(xTrain, yTrain)
baselineSkor = knnBaseline.score(xVal, yVal)
print(f"PCA olmadan (512 bit) geçerleme doğruluğı: %{baselineSkor*100:.2f}")

for pcaÖr in pcaBoyutları:

    if xTrain.shape[0] < pcaÖr or xTrain.shape[1] < pcaÖr:
        print(f"sistem uyarısı: Veri boyutu Pca {pcaÖr} için yetersiz")
        continue
    
    pca = PCA(n_components=pcaÖr)
    trainPca = pca.fit_transform(xTrain)
    xValPca = pca.transform(xVal)

    for k in kDeğerleri:
       knn = KNeighborsClassifier(n_neighbors=k)
       knn.fit(trainPca, yTrain)
       skor = knn.score(xValPca, yVal)

       print(f"PCA boyutu: {pcaÖr} | KNN K Değeri: {k} ==> Geçerleme Doğruluğu: %{skor*100:.2f}")

       if skor > enİyiSkor:
           enİyiSkor = skor
           enİyiPca = pcaÖr
           enİyiK = k
           enİyiModel = knn
           enİyiPcaObjesi = pca

print(f"\n[En iyi model] -> PCA: {enİyiPca}, K: {enİyiK}, val başarısı: %{enİyiSkor*100:.2f}")           

xTestPca = enİyiPcaObjesi.transform(xTest)
yPred = enİyiModel.predict(xTestPca)

print("\nTest Sonuçları")
print(classification_report(yTest, yPred))

print("\nConfusion Matrix")
print(confusion_matrix(yTest, yPred))
