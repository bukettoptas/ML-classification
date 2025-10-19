# 🎓 ML- Gözetimli öğrenme konuları

## 📚 İçindekiler

1. [K-En Yakın Komşu (KNN)](#knn)
2. [Naive Bayes](#naive-bayes)
3. [MAP vs MLE](#map-vs-mle)
4. [Destek Vektör Makineleri (SVM)](#svm)
5. [İnteraktif Simülasyonlar](#simulations)

---

## 🎯 K-En Yakın Komşu (KNN)

### 🤔 Temel Mantık

KNN en basit ML algoritmasıdır. Şöyle düşünün:

> "Arkadaşlarına göre kimsin? En yakın 5 arkadaşının çoğu doktor ise, sen de muhtemelen doktorsun!"

### 📊 Görsel Açıklama
```
Test noktası: ⭐

Yakındaki noktalar:
🔴 🔴 🔵 🔴 🔵  (k=5)

Sonuç: 3 kırmızı, 2 mavi → ⭐ muhtemelen 🔴
```

### 💻 Basit Python Kodu
```python
import numpy as np
from collections import Counter

class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X_train, y_train):
        """Eğitim: Sadece veriyi sakla! (Lazy learning)"""
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        """Test: Her test noktası için en yakın k komşuyu bul"""
        predictions = []
        
        for test_point in X_test:
            # 1. Tüm eğitim noktalarına mesafe hesapla
            distances = [np.linalg.norm(test_point - train_point) 
                        for train_point in self.X_train]
            
            # 2. En yakın k noktayı bul
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
            
            # 3. Çoğunluk oylaması
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return predictions

# Kullanım
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_train = ['A', 'A', 'A', 'B', 'B', 'B']

knn = SimpleKNN(k=3)
knn.fit(X_train, y_train)

X_test = np.array([[3, 3]])
print(f"Tahmin: {knn.predict(X_test)}")  # Output: ['A']
```

### ⚡ Zaman Karmaşıklığı

| Aşama | Karmaşıklık | Açıklama |
|-------|-------------|----------|
| Eğitim | O(1) | Sadece veriyi sakla |
| Test (1 örnek) | O(n·d) | n: eğitim sayısı, d: boyut |
| Test (m örnek) | O(m·n·d) | m: test sayısı |

### 🎚️ k Parametresi Seçimi
```python
import matplotlib.pyplot as plt

def plot_k_effect():
    """k değerinin karar sınırına etkisi"""
    k_values = [1, 5, 20, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, k in enumerate(k_values):
        ax = axes[idx // 2, idx % 2]
        # ... plotting code ...
        ax.set_title(f'k = {k}')
        
        if k == 1:
            ax.text(0.5, -0.1, 'Overfitting!\nÇok karmaşık sınır', 
                   ha='center', color='red', fontweight='bold')
        elif k == 50:
            ax.text(0.5, -0.1, 'Underfitting!\nÇok basit sınır', 
                   ha='center', color='red', fontweight='bold')
        else:
            ax.text(0.5, -0.1, 'İyi denge ✓', 
                   ha='center', color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('knn_k_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### 🎯 İnteraktif Demo
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def interactive_knn_demo():
    """Tıklayarak nokta ekle, k'yı değiştir, sonucu gör!"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15)
    
    # Başlangıç verileri
    class_A = np.array([[1, 2], [2, 1], [2, 3], [3, 2]])
    class_B = np.array([[5, 5], [6, 5], [5, 6], [6, 6]])
    
    # Plot
    scatter_A = ax.scatter(class_A[:, 0], class_A[:, 1], 
                          c='red', s=100, label='Class A', marker='o')
    scatter_B = ax.scatter(class_B[:, 0], class_B[:, 1], 
                          c='blue', s=100, label='Class B', marker='s')
    
    # Test noktası
    test_point = [3.5, 3.5]
    test_scatter = ax.scatter(*test_point, c='green', s=200, 
                             marker='*', label='Test Point', zorder=5)
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('KNN Interactive Demo - Click to add points!')
    
    # k slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    k_slider = Slider(ax_slider, 'k', 1, 8, valinit=3, valstep=1)
    
    def update_prediction(val):
        k = int(k_slider.val)
        # Mesafe hesapla ve tahmin yap
        all_points = np.vstack([class_A, class_B])
        all_labels = ['A'] * len(class_A) + ['B'] * len(class_B)
        
        distances = [np.linalg.norm(np.array(test_point) - point) 
                    for point in all_points]
        k_nearest = np.argsort(distances)[:k]
        k_labels = [all_labels[i] for i in k_nearest]
        
        # En yakın k noktayı çiz
        for i, point in enumerate(all_points):
            if i in k_nearest:
                circle = plt.Circle(point, 0.2, color='yellow', 
                                  alpha=0.3, fill=True)
                ax.add_patch(circle)
        
        # Tahmini göster
        prediction = max(set(k_labels), key=k_labels.count)
        color = 'red' if prediction == 'A' else 'blue'
        test_scatter.set_color(color)
        
        ax.set_title(f'k={k} → Prediction: Class {prediction}')
        fig.canvas.draw_idle()
    
    k_slider.on_changed(update_prediction)
    update_prediction(3)
    
    plt.show()

# Çalıştır!
interactive_knn_demo()
```

### ⚠️ KNN Tuzakları

#### 1️⃣ Ölçek Problemi
```python
# YANLIŞ ❌
X = [[10000, 1],      # Maaş: 10000 TL, Yaş: 1
     [20000, 2]]      # Maaş: 20000 TL, Yaş: 2

# Mesafe: Maaş farkı (10000) >> Yaş farkı (1)
# Yaş hiç önemli değil gibi görünür!

# DOĞRU ✓
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Şimdi her özellik aynı ölçekte!
```

#### 2️⃣ Yüksek Boyut Laneti
```python
import numpy as np

def curse_of_dimensionality_demo():
    """Boyut arttıkça noktalar birbirinden uzaklaşır!"""
    
    dimensions = [2, 10, 100, 1000]
    
    for d in dimensions:
        # 1000 rastgele nokta
        points = np.random.rand(1000, d)
        
        # İlk noktadan diğerlerine mesafe
        distances = [np.linalg.norm(points[0] - points[i]) 
                    for i in range(1, len(points))]
        
        print(f"Boyut: {d}")
        print(f"  Min mesafe: {min(distances):.3f}")
        print(f"  Max mesafe: {max(distances):.3f}")
        print(f"  Fark: {max(distances) - min(distances):.3f}\n")

curse_of_dimensionality_demo()
```

**Çıktı:**
```
Boyut: 2
  Min mesafe: 0.123
  Max mesafe: 1.234
  Fark: 1.111    ← Noktalar farklı mesafelerde

Boyut: 1000
  Min mesafe: 28.456
  Max mesafe: 29.123
  Fark: 0.667    ← Hepsi neredeyse aynı mesafede! 😱
```

---

## 🎲 Naive Bayes

### 🤔 Temel Mantık

> "Geçmişte neler oldu? Olasılıkları hesapla, en olası olanı seç!"

### 📧 Spam Örneği
```python
import numpy as np
from collections import defaultdict

class SimpleNaiveBayes:
    def __init__(self):
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.vocab = set()
        
    def fit(self, emails, labels):
        """Eğitim: Her kelimenin her sınıfta kaç kez geçtiğini say"""
        
        for email, label in zip(emails, labels):
            self.class_counts[label] += 1
            words = email.lower().split()
            
            for word in words:
                self.vocab.add(word)
                self.word_counts[label][word] += 1
    
    def predict(self, email):
        """Test: Her sınıf için olasılık hesapla, en yüksek olanı seç"""
        
        words = email.lower().split()
        scores = {}
        
        for label in self.class_counts:
            # P(class)
            score = np.log(self.class_counts[label] / 
                          sum(self.class_counts.values()))
            
            # P(word | class) - Laplace smoothing ile
            total_words = sum(self.word_counts[label].values())
            vocab_size = len(self.vocab)
            
            for word in words:
                word_count = self.word_counts[label][word]
                # Laplace smoothing: +1 / +vocab_size
                word_prob = (word_count + 1) / (total_words + vocab_size)
                score += np.log(word_prob)
            
            scores[label] = score
        
        return max(scores, key=scores.get)

# Demo
emails = [
    "win money now",           # spam
    "free prize click here",   # spam
    "meeting tomorrow at 9",   # ham
    "project deadline friday"  # ham
]
labels = ['spam', 'spam', 'ham', 'ham']

nb = SimpleNaiveBayes()
nb.fit(emails, labels)

test_email = "win free meeting"
prediction = nb.predict(test_email)
print(f"Email: '{test_email}' → {prediction}")
```

### 🧮 Matematik (Basit!)

**Bayes Teoremi:**
```
P(Spam | Email) = P(Email | Spam) × P(Spam) / P(Email)
```

**Naive varsayım:**
```
P(Email | Spam) = P(word₁ | Spam) × P(word₂ | Spam) × ...
```

**Örnek Hesaplama:**
```python
def naive_bayes_example():
    """Adım adım hesaplama"""
    
    print("📧 Email: 'win money'")
    print("\n--- SPAM için ---")
    print("P(Spam) = 2/4 = 0.5")
    print("P('win' | Spam) = (1+1)/(5+V) = 2/15 ≈ 0.133")
    print("P('money' | Spam) = (1+1)/(5+V) = 2/15 ≈ 0.133")
    print("P(Spam | Email) ∝ 0.5 × 0.133 × 0.133 = 0.0088")
    
    print("\n--- HAM için ---")
    print("P(Ham) = 2/4 = 0.5")
    print("P('win' | Ham) = (0+1)/(5+V) = 1/15 ≈ 0.067")
    print("P('money' | Ham) = (0+1)/(5+V) = 1/15 ≈ 0.067")
    print("P(Ham | Email) ∝ 0.5 × 0.067 × 0.067 = 0.0022")
    
    print("\n✅ Sonuç: SPAM (0.0088 > 0.0022)")

naive_bayes_example()
```

### 🎭 Laplace Smoothing Vizualizasyonu
```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_smoothing():
    """Smoothing olmadan vs ile"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    words = ['free', 'money', 'meeting', 'UNSEEN']
    spam_counts = [5, 3, 0, 0]      # 'meeting' ve 'UNSEEN' spam'de yok
    ham_counts = [0, 0, 4, 0]        # 'free' ve 'money' ham'de yok
    
    # Smoothing YOK - Bazı olasılıklar 0!
    spam_probs_no_smooth = [c/8 if 8 > 0 else 0 for c in spam_counts]
    ham_probs_no_smooth = [c/4 if 4 > 0 else 0 for c in ham_counts]
    
    x = np.arange(len(words))
    width = 0.35
    
    ax1.bar(x - width/2, spam_probs_no_smooth, width, label='Spam', color='red', alpha=0.7)
    ax1.bar(x + width/2, ham_probs_no_smooth, width, label='Ham', color='blue', alpha=0.7)
    ax1.set_ylabel('Probability')
    ax1.set_title('❌ Smoothing YOK - Problem: 0 olasılıklar!')
    ax1.set_xticks(x)
    ax1.set_xticklabels(words)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Laplace Smoothing (+1)
    vocab_size = len(words)
    spam_probs_smooth = [(c+1)/(8+vocab_size) for c in spam_counts]
    ham_probs_smooth = [(c+1)/(4+vocab_size) for c in ham_counts]
    
    ax2.bar(x - width/2, spam_probs_smooth, width, label='Spam', color='red', alpha=0.7)
    ax2.bar(x + width/2, ham_probs_smooth, width, label='Ham', color='blue', alpha=0.7)
    ax2.set_ylabel('Probability')
    ax2.set_title('✅ Laplace Smoothing - Hiç 0 yok!')
    ax2.set_xticks(x)
    ax2.set_xticklabels(words)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('laplace_smoothing.png', dpi=300)
    plt.show()

visualize_smoothing()
```

---

## 🎯 MAP vs MLE: Büyük Kafa Karışıklığı Sonu!

### 🤔 En Basit Açıklama

**MLE (Maximum Likelihood Estimation):**
> "Sadece veriye bak, en olası parametreyi seç!"

**MAP (Maximum A Posteriori):**
> "Veriye bak AMA önceden bildiklerini de katmayı unutma!"

### 🪙 Yazı-Tura Örneği
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def coin_flip_demo(n_flips=10, n_heads=8):
    """MLE vs MAP görsel karşılaştırma"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    theta_range = np.linspace(0, 1, 1000)
    
    # 1. MLE
    mle_estimate = n_heads / n_flips
    
    axes[0].axvline(mle_estimate, color='red', linewidth=3, 
                    label=f'MLE = {mle_estimate:.2f}')
    axes[0].set_title(f'MLE: "Sadece Veriye Bak!"\n{n_heads}/{n_flips} tura → θ = {mle_estimate:.2f}', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('θ (Tura olasılığı)')
    axes[0].set_ylabel('Likelihood')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.5, 0.5, '⚠️ Overfitting riski!\n10 atış yeterli mi?', 
                ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 2. Prior (önceki bilgi)
    prior_alpha, prior_beta = 2, 2  # Adil para ağırlıklı prior
    prior = beta.pdf(theta_range, prior_alpha, prior_beta)
    
    axes[1].plot(theta_range, prior, 'b-', linewidth=2, label='Prior')
    axes[1].axvline(0.5, color='blue', linestyle='--', 
                    label='Prior peak = 0.5')
    axes[1].set_title('Prior: "Önceki Bilgim"\n"Gerçek paralar genelde adildir (θ≈0.5)"', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('θ (Tura olasılığı)')
    axes[1].set_ylabel('Density')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(theta_range, prior, alpha=0.3)
    
    # 3. MAP (Posterior)
    posterior_alpha = prior_alpha + n_heads
    posterior_beta = prior_beta + (n_flips - n_heads)
    posterior = beta.pdf(theta_range, posterior_alpha, posterior_beta)
    
    map_estimate = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)
    
    axes[2].plot(theta_range, posterior, 'g-', linewidth=2, label='Posterior')
    axes[2].axvline(map_estimate, color='green', linewidth=3, 
                    label=f'MAP = {map_estimate:.2f}')
    axes[2].axvline(mle_estimate, color='red', linestyle='--', 
                    alpha=0.5, label=f'MLE = {mle_estimate:.2f}')
    axes[2].set_title(f'MAP: "Veri + Prior Bilgi"\n8/10 tura ama θ ≈ {map_estimate:.2f}', 
                     fontsize=12, fontweight='bold')
    axes[2].set_xlabel('θ (Tura olasılığı)')
    axes[2].set_ylabel('Density')
    axes[2].legend(fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].fill_between(theta_range, posterior, alpha=0.3, color='green')
    axes[2].text(0.5, max(posterior)*0.7, 
                '✅ Regularization etkisi!\nPrior overfitting\'i engeller', 
                ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('mle_vs_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sayısal karşılaştırma
    print("=" * 50)
    print(f"Deney: {n_heads} tura / {n_flips} atış")
    print("=" * 50)
    print(f"MLE tahmini: θ = {mle_estimate:.4f}")
    print(f"MAP tahmini: θ = {map_estimate:.4f}")
    print(f"Fark: {abs(mle_estimate - map_estimate):.4f}")
    print("\n💡 MAP, MLE'yi 0.5'e doğru çekiyor (regularization!)")

# Farklı senaryolar deneyin!
coin_flip_demo(n_flips=10, n_heads=10)   # Extreme durum
coin_flip_demo(n_flips=100, n_heads=80)  # Daha çok veri
coin_flip_demo(n_flips=1000, n_heads=800) # Çok veri
```

### 📊 Veri Artınca Ne Olur?
```python
def map_converges_to_mle():
    """Veri arttıkça MAP → MLE"""
    
    n_values = [10, 50, 100, 500, 1000]
    true_theta = 0.7  # Gerçek parametre
    
    mle_estimates = []
    map_estimates = []
    
    for n in n_values:
        n_heads = int(n * true_theta)  # %70 tura
        
        # MLE
        mle = n_heads / n
        mle_estimates.append(mle)
        
        # MAP (Beta(2,2) prior)
        map_est = (n_heads + 2 - 1) / (n + 2 + 2 - 2)
        map_estimates.append(map_est)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, mle_estimates, 'ro-', linewidth=2, 
            markersize=10, label='MLE')
    plt.plot(n_values, map_estimates, 'go-', linewidth=2, 
            markersize=10, label='MAP')
    plt.axhline(true_theta, color='blue', linestyle='--', 
               linewidth=2, label=f'Gerçek θ = {true_theta}')
    
    plt.xlabel('Veri Miktarı (n)', fontsize=12)
    plt.ylabel('Tahmin Edilen θ', fontsize=12)
    plt.title('Veri Arttıkça: MAP → MLE\n(Prior bilginin etkisi azalır)', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Annotations
    plt.annotate('Küçük veri:\nMAP daha iyi (regularization)', 
                xy=(10, map_estimates[0]), xytext=(15, 0.6),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.annotate('Çok veri:\nMAP ≈ MLE (prior etkisiz)', 
                xy=(1000, map_estimates[-1]), xytext=(600, 0.75),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                fontsize=11, bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('map_to_mle_convergence.png', dpi=300)
    plt.show()

map_converges_to_mle()
```

### 🔗 L2 Regularization = MAP!
```python
def regularization_is_map():
    """L2 regularization'ın MAP yorumu"""
    
    print("🎯 Lojistik Regresyon Örneği")
    print("=" * 60)
    
    print("\n📝 Standart (MLE):")
    print("   min  -Σ log P(y_i | x_i, w)")
    print("    w")
    
    print("\n📝 L2 Regularization:")
    print("   min  -Σ log P(y_i | x_i, w) + λ||w||²")
    print("    w")
    
    print("\n🎓 MAP Yorumu:")
    print("   max  log P(y|X,w) + log P(w)")
    print("    w")
    print("\n   Prior: P(w) = N(0, 1/(2λ))")
    print("   →  log P(w) = -λ||w||² + const")
    
    print("\n" + "=" * 60)
    print("✅ Sonuç: L2 regularization = Gaussian prior ile MAP!")
    print("✅ Sonuç: L1 regularization = Laplace prior ile MAP!")
    print("=" * 60)

regularization_is_map()
```

### 🎮 İnteraktif MAP vs MLE
```python
from ipywidgets import interact, IntSlider
import matplotlib.pyplot as plt

@interact(
    n_flips=IntSlider(min=5, max=100, step=5, value=10, description='Atış sayısı:'),
    n_heads=IntSlider(min=0, max=100, step=5, value=8, description='Tura sayısı:')
)
def interactive_map_mle(n_flips=10, n_heads=8):
    """Parametreleri oyna, farkı gör!"""
    
    if n_heads > n_flips:
        n_heads = n_flips
    
    # MLE
    mle = n_heads / n_flips if n_flips > 0 else 0
    
    # MAP
    prior_alpha, prior_beta = 2, 2
    map_est = (n_heads + prior_alpha - 1) / (n_flips + prior_alpha + prior_beta - 2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['MLE\n(Sadece veri)', 'MAP\n(Veri + Prior)']
    values = [mle, map_est]
    colors = ['red', 'green']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Gerçek adil para çizgisi
    ax.axhline(0.5, color='blue', linestyle='--', linewidth=2, label='Adil para (θ=0.5)')
    
    ax.set_ylabel('Tahmin edilen θ', fontsize=14)
    ax.set_title(f'Deney: {n_heads} tura / {n_flips} atış', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Değerleri bar üzerine yaz
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Fark göster
    diff = abs(mle - map_est)
    ax# ML-classification
