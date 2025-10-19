# 🎯 K-En Yakın Komşu (KNN)

[Ana Sayfaya Dön](../README.md)

## 🤔 Temel Mantık

KNN en basit ML algoritmasıdır. Şöyle düşünün:

> "Arkadaşlarına göre kimsin? En yakın 5 arkadaşının çoğu doktor ise, sen de muhtemelen doktorsun!"

### 📊 Görsel Açıklama

Aşağıdaki görselde, ortadaki test noktasının (⭐) çevresindeki en yakın 5 komşu (k=5) baz alınarak hangi sınıfa ait olduğuna karar verildiğini görebilirsiniz. Çoğunluk kırmızı olduğu için test noktası da kırmızı olarak tahmin edilir.

http://googleusercontent.com/image_generation_content/0



### 💻 Basit Python Kodu

Algoritmanın basit bir Python implementasyonu için `simple_knn.py` dosyasına bakabilirsiniz.

### ⚡ Zaman Karmaşıklığı

| Aşama | Karmaşıklık | Açıklama |
| --- | --- | --- |
| Eğitim | $O(1)$ | Sadece veriyi sakla (Lazy Learning) |
| Test (1 örnek) | $O(n \cdot d)$ | n: eğitim sayısı, d: özellik boyutu |
| Test (m örnek) | $O(m \cdot n \cdot d)$ | m: test sayısı |

### 🎯 İnteraktif Simülasyonlar

Algoritmanın nasıl çalıştığını görmek, `k` değerini değiştirmek ve "Yüksek Boyut Laneti"ni görselleştirmek için `knn_simulations.ipynb` dosyasını çalıştırın.

### ⚠️ KNN Tuzakları

#### 1️⃣ Ölçek Problemi (Feature Scaling)

KNN, mesafeye dayalı bir algoritmadır. Eğer özellikleriniz farklı ölçeklerdeyse (örn: Maaş [10.000 - 100.000] ve Yaş [20-60]), maaş özelliği mesafeyi domine edecektir.

**Çözüm:** `StandardScaler` (Z-skoru) veya `MinMaxScaler` (0-1 arasına sıkıştırma) gibi özellik ölçeklendirme teknikleri **mutlaka** kullanılmalıdır.

```python
# YANLIŞ ❌
# Maaş: 10000, Yaş: 1
# Maaş: 20000, Yaş: 2
# Mesafe: Maaş farkı (10000) >> Yaş farkı (1)
# Yaş özelliği neredeyse hiç etki etmez!

# DOĞRU ✓
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Şimdi her özellik aynı ölçekte (ortalama=0, std=1)
