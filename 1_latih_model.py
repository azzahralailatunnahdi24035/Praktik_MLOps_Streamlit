
import joblib
from sklearn.tree import DecisionTreeClassifier
import numpy as np

print("Sedang melatih model AI...")

# Data latih: [IPK, Total SKS]
X_train = np.array([
    [3.5, 144],
    [2.0, 100],
    [3.2, 140],
    [2.5, 110],
    [3.8, 150]
])

# Label: 1 = Lulus, 0 = Tidak Lulus
y_train = np.array([1, 0, 1, 0, 1])

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, 'model_kelulusan.pkl')

print("Sukses! Model disimpan sebagai 'model_kelulusan.pkl'")
