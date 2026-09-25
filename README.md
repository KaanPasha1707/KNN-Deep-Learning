# AU-AIR Veri Seti ile İHA Görüntülerinde Nesne Sınıflandırma

Bu proje, AU-AIR veri setindeki insansız hava aracı (İHA/Drone) görüntüleri üzerinde yer alan nesneleri, makine öğrenmesi ve derin öğrenme tekniklerini harmanlayarak sınıflandırmayı amaçlamaktadır. Görüntülerdeki nesneler önceden eğitilmiş (pre-trained) derin öğrenme modelleri kullanılarak vektör uzayına dönüştürülmüş ve KNN algoritması ile sınıflandırılmıştır.

**AU-AIR Veri Seti Görselleri:** https://drive.google.com/open?id=1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ 
**AU-AIR Veri Seti Etiketleri:** https://drive.google.com/open?id=1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA

## 🏗️ Proje Mimarisi ve İş Akışı

Sistem temel olarak dört aşamadan oluşmaktadır:

1. **Veri Ön İşleme (Kırpma):** `annotations.json` dosyasındaki koordinatlar (bounding box) kullanılarak her bir nesne (araba, insan vb.) resimden tek tek kırpılmış ve arka plan gürültüsü izole edilmiştir.
2. **Öznitelik Çıkarımı (Feature Extraction):** Kırpılan resimler `img2vec_pytorch` kütüphanesi yardımıyla ResNet18 (512 boyut) ve AlexNet (4096 boyut) modellerine beslenerek matematiksel öznitelik vektörleri elde edilmiştir.
3. **Boyut İndirgeme (PCA):** Çıkarılan yüksek boyutlu vektörler, sistem kaynaklarını korumak ve KNN hesaplama yükünü hafifletmek amacıyla PCA kullanılarak 10, 20 ve 30 boyuta düşürülmüştür.
4. **Sınıflandırma:** Boyutu düşürülmüş veriler üzerinde K=1, 3 ve 5 değerleri denenerek KNN modeli eğitilmiş ve test edilmiştir.

## 📊 Veri Seti Dağılımı

Modelin aynı resimdeki nesneleri ezberlemesini (veri sızıntısı) önlemek amacıyla resim bazlı %70, %15 ve %15 bölünme uygulanmıştır:
*   **Eğitim (Train):** 92.493 nesne.
*   **Geçerleme (Validation):** 19.864 nesne.
*   **Test:** 19.620 nesne.

Veri seti toplam 8 sınıftan oluşmaktadır (Human, Car, Truck, Van, Motorbike, Bicycle, Bus, Trailer). Eğitim setinin çok büyük bir kısmı "Car" (Araba) sınıfından oluştuğu için veri setinde yüksek oranda sınıf dengesizliği (class imbalance) mevcuttur.

## 🏆 Deneysel Sonuçlar ve En İyi Model

Hiperparametre optimizasyonu sürecinde farklı PCA bileşen sayıları ve KNN komşuluk değerleri (K) test edilmiştir.

*   **Temel Kontrol (Baseline):** AlexNet vektörleri boyut azaltma (PCA) işlemi uygulanmadan doğrudan KNN'e verildiğinde %89.19 geçerleme doğruluğu elde edilmiştir.
*   **En İyi Hiperparametreler:** En yüksek başarı oranını sağlayan kombinasyon **AlexNet modeli, PCA boyutu: 30 ve KNN (K=5)** olarak belirlenmiştir.
*   **Nihai Başarı:** Bu optimizasyon sayesinde girdi verisi %99.2 oranında sıkıştırılmasına rağmen doğruluk oranında çok küçük bir kayıp yaşanmış ve nihai test setinde **%89.00** genel doğruluk oranına ulaşılmıştır.

## ⚙️ Kurulum ve Çalıştırma

**Gereksinimler:**
Projenin çalışması için `numpy`, `Pillow`, `scikit-learn`, `img2vec_pytorch` ve `matplotlib` kütüphanelerinin yüklü olması gerekmektedir.

1. Depoyu bilgisayarınıza klonlayın.
2. Gerekli bağımlılıkları sisteminize kurun.
3. `VektörBulma.py` dosyasını çalıştırarak kırpma ve model eğitim sürecini başlatın.

> **⚠️ Önemli Not (Veri Boyutu):**
> Vektör çıkarımı sonucunda oluşan devasa boyutlu matris dosyaları (`.npy`), GitHub'ın 100 MB tekil dosya sınırını aştığı için bu depoya yüklenmemiştir. `VektörBulma.py` dosyası çalıştırıldığında, resimlerden öznitelikleri çıkararak bu `.npy` dosyalarını bilgisayarınızda lokal olarak otomatik üretecek ve kaydedecektir.
> İndireceğiniz dosyaların isimleri koddakinden farklı olabilir. O isimlere göre kodu güncellemeyi unutmayın. 
