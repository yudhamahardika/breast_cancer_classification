# breast_cancer_classification
Perbandingan Logistic Regression dan KNN untuk klasifikasi kanker payudara menggunakan dataset Wisconsin Breast Cancer Diagnostic

## Tujuan
Proyek ini bertujuan untuk membandingkan performa dua algoritma Machine Learning,
yaitu *Logistic Regression* dan *K-Nearest Neighbors (KNN)*, 
dalam mendeteksi kanker payudara menggunakan dataset 
[Wisconsin Breast Cancer Diagnostic](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

## Dataset
- Sumber: Kaggle
- Jumlah data: 569 sampel
- Label: 
  - M (Malignant → kanker)
  - B (Benign → tidak kanker)

## Metodologi
1. Data preprocessing (cek missing values, scaling menggunakan MinMaxScaler & StandardScaler).
2. Model training dengan:
   - Logistic Regression
   - KNN (dengan GridSearchCV untuk memilih k terbaik).
3. Evaluasi menggunakan *accuracy score* dan *confusion matrix*.

## Hasil
- Logistic Regression: akurasi 98%
- KNN: akurasi 97%
- Visualisasi perbandingan akurasi juga ditampilkan dalam grafik.

## Teknologi
- Python
- Pandas, Numpy
- Scikit-learn
- Matplotlib, Seaborn

## Kesimpulan
Logistic Regression lebih unggul daripada KNN untuk dataset ini (hasil sesuai eksperimen). 
Proyek ini bisa menjadi baseline untuk penelitian lebih lanjut di bidang *healthcare AI*.