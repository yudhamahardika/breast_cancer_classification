from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.pipeline import Pipeline 


# load data
data = pd.read_csv("brca.csv")
df = pd.DataFrame(data)

print("Info data: \n")
print(df.info())
print("Describe data: \n", df.describe())
print("Menampilkan 5 dataset: \n", df.head())
print(df.isnull().sum())
print(df.columns)

# Hapus kolonm yang tidak berguna
df = df.drop(columns=["Unnamed: 0"])

# pemisahan feature dan target
x = df.drop(columns=["y"])
y = df["y"]

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Membuat pipeline Logistic Regression
pipe_logre = Pipeline([
    ("scaler", StandardScaler()),
    ("logre", LogisticRegression(max_iter=500))
])

# Hyperparameter Logistic Regression
param_logre = {
    "scaler": [StandardScaler(), MinMaxScaler()],
    "logre__C": [0.01, 0.1, 1, 10],
    "logre__solver": ["lbfgs", "liblinear"]
}

grid_logre = GridSearchCV(pipe_logre, param_logre, cv=5, scoring="accuracy")
grid_logre.fit(x_train, y_train)

print("Best Logistic Regression Param: \n", grid_logre.best_params_)
print("Best Logistic Regression Score: \n", grid_logre.best_score_)

# Membuat pipeline KNN
pipe_knn = Pipeline([
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsClassifier())
])

# Hyperparameter KNN
param_knn = {
    "scaler": [StandardScaler(), MinMaxScaler()],
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"]
}

grid_knn = GridSearchCV(pipe_knn, param_knn, cv=5, scoring="accuracy")
grid_knn.fit(x_train, y_train)

print("Best KNN_Params: ", grid_knn.best_params_)
print("Best KNN Score: ", grid_knn.best_score_)

# Evaluasi terbaik
y_pred_logre = grid_logre.best_estimator_.predict(x_test)
y_pred_knn = grid_knn.best_estimator_.predict(x_test)

print("\n Logistic Regression Evaluation")
print(confusion_matrix(y_test, y_pred_logre))
print(classification_report(y_test, y_pred_logre))
print("ROC-AUC Logistic Regression: ", roc_auc_score(y_test, grid_logre.predict_proba(x_test)[:, 1]))

print("\n KNN Evaluation")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print("ROC-AUC KNN: ", roc_auc_score(y_test, grid_knn.predict_proba(x_test)[:, 1]))

# Visualisasi Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_logre)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn)
plt.title("KNN Confusion Matrix")
plt.show()

# Visualisasi akurasi per model
score = {
    "Logistic Regression": accuracy_score(y_test, y_pred_logre),
    "KNN": accuracy_score(y_test, y_pred_knn)
}

sns.barplot(x=list(score.keys()), y=list(score.values()))
plt.title("Perbandingan Akurasi MOdel")
plt.show()