# ML Foundations from Scratch — 7 Günlük Yol Haritası

> Andrew Ng ML Specialization C1 W1+W2 lab'larından üretilmiş, NumPy ile sıfırdan
> lineer regresyon inşa eden 7 günlük pratik projesi.
>
> **Amaç:** Teoriyi kodla pekiştirmek, her günün sonunda çalışan + commit'lenmiş
> bir parça çıkarmak. Mükemmel değil — çalışan ve anlaşılır.

---

## Repo Yapısı

```
ml-foundations-from-scratch/
├── README.md
├── ROADMAP.md                      ← bu dosya
├── data/
│   └── houses.txt                  ← Andrew Ng'nin 47 örnek house dataset
├── src/
│   ├── day01_numpy_warmup.py
│   ├── day02_cost_function.py
│   ├── day03_gradient_descent.py
│   ├── day04_multivariate.py
│   ├── day05_feature_scaling.py
│   ├── day06_polynomial_features.py
│   └── day07_sklearn_compare.py
├── plots/                          ← her gün üretilen .png'ler buraya
└── notes/
    └── dayXX_notes.md              ← her gün ne öğrendin (kısa)
```

---

## Genel Kurallar

1. **Her gün 1 commit minimum.** Mesaj formatı: `Day X: <ne yaptım>` (örn. `Day 2: implement cost function with vectorized version`).
2. **Her dosyanın başında docstring:** ne yapıyor, hangi denklem, hangi veri.
3. **Her dosya `if __name__ == "__main__":` ile çalışabilir olmalı** — `python src/day02_cost_function.py` çalıştırınca sonuç versin.
4. **Plot üreten her gün → `plots/dayXX_*.png` olarak kaydet** (`plt.savefig`).
5. **Kabul kriteri tutmuyorsa o gün bitmez.** Ama mükemmellik aranmaz — çalışan ve doğrulanmış yeter.

---

## Day 1 — NumPy Isınma + Repo Setup

**Hedef:** NumPy'nin lineer regresyonda kullanılan temel işlemlerini parmak hafızasına almak. Vektör/matris zihninde otursun.

**Öğrenme Konuları**
- `np.array`, `shape`, `ndim`, `dtype`
- Vektör + skaler, vektör + vektör (broadcasting)
- `np.dot`, `@` operatörü, `X.T` (transpoz)
- `np.zeros`, `np.ones`, `np.linspace`, `np.arange`
- Indexleme: `X[:, 0]`, `X[i, :]`, slicing

**Egzersizler (`src/day01_numpy_warmup.py`)**
1. `x = np.array([1.0, 2.0, 3.0, 4.0])` ve `y = np.array([2.0, 4.0, 6.0, 8.0])` oluştur. Shape'lerini print et.
2. `w = 2`, `b = 0` için `f = w * x + b` hesapla. Tahminin gerçekle eşleştiğini print ile doğrula.
3. 3x2'lik bir `X` matrisi yarat. `X.T @ X` ve `X @ X.T` boyutlarını print et — neden farklı?
4. `np.dot(x, y)` ile manuel `sum(x[i]*y[i] for i in range(len(x)))` aynı sonucu mu veriyor? Karşılaştır.
5. `data/houses.txt`'i `np.loadtxt(..., delimiter=',')` ile yükle. Shape'i, ilk 5 satırı print et.

**Kabul Kriteri**
- Dosya hatasız çalışıyor.
- 5 egzersizin print çıktıları doğru.
- Repo GitHub'da, README'de proje açıklaması var.

**Commit:** `Day 1: NumPy warmup + repo setup`

---

## Day 2 — Cost Function (Univariate)

**Hedef:** `J(w,b)` cost function'ını anlayıp sıfırdan yazmak. Cost'un parametreye göre nasıl değiştiğini görselleştirmek.

**Teori (kısa)**

```
f(x) = w*x + b
J(w,b) = (1 / (2m)) * sum( (f(x_i) - y_i)^2 )
```

m = örnek sayısı. 2 bölüm türev temizliği için.

**Egzersizler (`src/day02_cost_function.py`)**
1. `compute_cost(x, y, w, b)` fonksiyonunu yaz — **for loop ile**.
2. `compute_cost_vectorized(x, y, w, b)` fonksiyonunu yaz — **NumPy vektörize**. İkisi aynı sonucu vermeli (`np.allclose` ile doğrula).
3. Lab'daki örneği kullan: `x = [1.0, 2.0]`, `y = [300.0, 500.0]`. `w=200, b=100` için cost = 0 olmalı. Doğrula.
4. `b = 100` sabit. `w` için 0'dan 400'e 100 değer dene, her biri için cost hesapla. `cost vs w` grafiğini çiz, kaydet (`plots/day02_cost_vs_w.png`).
5. Lab'daki büyük datasette (`x = [1.0, 1.7, 2.0, 2.5, 3.0, 3.2]`, `y = [250, 300, 480, 430, 630, 730]`) `w=209, b=2.4` ile cost'u hesapla. Sıfır mı? Neden değil?

**Kabul Kriteri**
- Loop ve vektörize versiyonlar aynı sonucu veriyor.
- `cost vs w` grafiği üretildi, minimum w=200 civarında.
- 5. egzersizin yorumu `notes/day02_notes.md`'de yazılı (2-3 cümle).

**Commit:** `Day 2: implement cost function (loop + vectorized) with visualization`

---

## Day 3 — Gradient Descent (Univariate)

**Hedef:** Cost'u manuel optimize etmek yerine türev kullanarak otomatik minimuma indirmek. "Bilgisayar nasıl öğrenir"in özü.

**Teori (kısa)**

```
dJ/dw = (1/m) * sum( (f(x_i) - y_i) * x_i )
dJ/db = (1/m) * sum(  f(x_i) - y_i )

w := w - α * dJ/dw
b := b - α * dJ/db
```

α = learning rate.

**Egzersizler (`src/day03_gradient_descent.py`)**
1. `compute_gradient(x, y, w, b)` yaz → `(dj_dw, dj_db)` döndür.
2. `gradient_descent(x, y, w_init, b_init, alpha, num_iters)` yaz. Her iterasyonda `w, b, J` değerini bir history listesine kaydet.
3. Day 2'deki küçük datasetle başla (`[1.0, 2.0]`, `[300, 500]`). `w=0, b=0`, `α=0.01`, 10000 iter. Final `w, b` 200 ve 100'e yakın olmalı.
4. `J` history'sini iterasyona karşı çiz (`plots/day03_loss_curve.png`). Düşmüyor mu? α'yı küçült.
5. **α deneyleri:** `α = 0.1, 0.01, 0.001` için 1000 iter koş. Hangisinde patlıyor (NaN/inf), hangisinde yavaş, hangisinde sağlam? `notes/day03_notes.md`'ye yaz.

**Kabul Kriteri**
- Gradient descent yakınsıyor (final cost < 1.0 küçük dataset için).
- Loss curve monotonik düşüyor (tepe yok).
- α deneylerinin gözlemleri yazılı.

**Commit:** `Day 3: gradient descent with learning rate experiments`

---

## Day 4 — Multivariate Linear Regression

**Hedef:** Tek feature'dan çoklu feature'a geçmek. Vektörizasyonun gerçek faydasını görmek (`X @ w`).

**Teori (kısa)**

```
f(x) = w1*x1 + w2*x2 + ... + wn*xn + b = X @ w + b
```

X shape: `(m, n)`. w shape: `(n,)`. b skaler.

**Egzersizler (`src/day04_multivariate.py`)**
1. `data/houses.txt`'i yükle. İlk 2 sütun feature (size, bedrooms), 3. sütun fiyat. `X` ve `y` olarak ayır. Shape'leri print et.
2. `compute_cost_multi(X, y, w, b)` yaz — vektörize, `X @ w + b` kullan.
3. `compute_gradient_multi(X, y, w, b)` yaz — vektörize, `X.T @ error` kullan.
4. `gradient_descent_multi(...)` yaz (Day 3'tekinin multi-feature versiyonu).
5. `w_init = np.zeros(n)`, `b_init = 0`, `α = 1e-7` (dikkat — feature'lar büyük), 1000 iter koş. Yakınsıyor mu?
6. Loss curve çiz (`plots/day04_loss_curve.png`). Düşmüyorsa neden? (Spoiler: feature scaling lazım — Day 5.)

**Kabul Kriteri**
- Kod hatasız çalışıyor.
- Loss curve elde edildi (yakınsamayabilir, sorun değil — Day 5'te çözülecek).
- `notes/day04_notes.md`'de "neden α bu kadar küçük olmak zorunda kaldı?" sorusunun cevabı yazılı.

**Commit:** `Day 4: multivariate linear regression on house dataset`

---

## Day 5 — Feature Scaling + Learning Rate

**Hedef:** Feature'ları aynı skalaya getirip gradient descent'i hızlandırmak. Day 4'ün acısını dindirmek.

**Teori (kısa)**

Z-score normalization:
```
x_scaled = (x - μ) / σ
```

Her feature için ayrı μ ve σ. Sonra büyük α (örn. 0.1) kullanılabilir.

**Egzersizler (`src/day05_feature_scaling.py`)**
1. `zscore_normalize(X)` yaz → `X_norm, mu, sigma` döndür. (μ, σ'yı tutmak önemli — yeni veri için aynısını uygulayacaksın.)
2. Day 4'teki house verisinin X'ini normalize et. Her feature için μ, σ ve normalize sonrası min/max'ı print et.
3. Day 4'teki gradient descent'i bu sefer `X_norm` ile çalıştır. `α = 0.1`, 1000 iter. Final cost ve `w, b`'yi print et.
4. `α = 0.01, 0.1, 1.0` için ayrı ayrı koş, loss curve'leri **aynı grafikte** çiz (`plots/day05_alpha_compare.png`).
5. Yeni bir ev tahmini yap: 1200 sqft, 3 yatak odası. **Aynı μ, σ ile normalize et**, sonra `np.dot(x_norm, w) + b` ile tahmin et. Sonucu print et.

**Kabul Kriteri**
- α=0.1 ile 1000 iter'da rahat yakınsıyor.
- 3 farklı α'nın karşılaştırma grafiği üretildi.
- Yeni veri tahmini çalışıyor — `notes/day05_notes.md`'de "scaling olmadan ne olurdu?" notu var.

**Commit:** `Day 5: z-score normalization unlocks faster convergence`

---

## Day 6 — Polynomial Features

**Hedef:** Lineer modelle non-lineer ilişkiyi yakalamak. Feature engineering'in özü.

**Teori (kısa)**

Lineer model lineer kalır ama feature'ları dönüştürürsen (`x → x², x³, vs.`) non-lineer ilişki yakalarsın.
```
f(x) = w1*x + w2*x² + w3*x³ + b
```

**Egzersizler (`src/day06_polynomial_features.py`)**
1. Sentetik veri üret: `x = np.arange(0, 20, 1)`, `y = x**2 + np.random.randn(20)*5`. Scatter plot çiz.
2. Önce lineer modelle (sadece `x` feature'ı) Day 5 pipeline'ını koş. Tahmini gerçek üzerine çiz — kötü olacak, görmek için çiz.
3. `X_poly = np.c_[x, x**2, x**3]` ile polynomial feature matrisi oluştur. Z-score normalize et.
4. Bu `X_poly` ile gradient descent koş. `w` katsayılarını print et — hangi feature en büyük ağırlığı aldı? (x² olmalı.)
5. Tahmini gerçek üzerine çiz (`plots/day06_polynomial_fit.png`). Lineer vs polinom — yan yana subplot.

**Kabul Kriteri**
- Polinom modeli `x²`'yi yakaladı (`w[1]` en büyük).
- Lineer vs polinom karşılaştırma grafiği üretildi.
- `notes/day06_notes.md`: "Hangi feature engineering kararını verdim ve neden?"

**Commit:** `Day 6: polynomial features capture non-linear patterns`

---

## Day 7 — Scikit-learn ile Karşılaştırma + Final README

**Hedef:** Sıfırdan yazdığın şeyi 3 satırda yapan kütüphaneyi öğrenmek. İkisinin sonucunu karşılaştırıp aradaki farkı anlamlandırmak.

**Egzersizler (`src/day07_sklearn_compare.py`)**
1. `from sklearn.linear_model import SGDRegressor` ve `from sklearn.preprocessing import StandardScaler`.
2. House verisini yükle, `StandardScaler` ile normalize et, `SGDRegressor(max_iter=1000)` ile fit et.
3. `sgdr.coef_` ve `sgdr.intercept_`'i Day 5'teki kendi `w, b`'nle yan yana print et. Yakın mı?
4. `LinearRegression` (closed-form / normal equation) ile aynısını yap. Sonuçları üçlü karşılaştır.
5. Bir tahmin: 1200 sqft, 3 yatak — kendi modelinle, SGDRegressor ile, LinearRegression ile. 3 sonucu karşılaştır.
6. **README.md'yi yaz:** her günün ne yaptığını, hangi grafikleri ürettiğini özetle. Plot'ları markdown'a embed et.

**Kabul Kriteri**
- 3 modelin tahminleri yakın (≈%5 fark içinde).
- README cilalı: ne yapıldı, nasıl çalıştırılır, sonuçlar.
- Repo "show-able" — birine link atabilirsin.

**Commit:** `Day 7: sklearn comparison + final README`

---

## Bitince Sende Olan

- 7 commit'lik bir GitHub repo'su
- Her ML fundamental'i bizzat kodlamış olmanın güveni
- Lineer regresyonun **tüm parçalarını** (cost, gradient, vektörizasyon, scaling, feature engineering, library kullanımı) bilmek
- Bir sonraki adım için sağlam zemin: **logistic regression** (aynı pipeline, sigmoid + log loss değişikliğiyle)

---

## Eğer Sıkışırsan

- Bilgi karmaşası başlarsa → SADECE o günün dosyasını aç, başka kaynak açma.
- "Yeterince anlamadım" hissi → o günü bitir, diğerine geç. Anlama bir sonraki günde gelir.
- Bir gün kaçırırsan → ertesi gün yine bir gün, panik yok. Ama 2 gün üst üste kaçırma.
