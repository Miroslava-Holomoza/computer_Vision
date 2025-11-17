import pandas as pd #для роботи з даними у форматі таблиць
import numpy as np #для роботи з масивами та таблицями
import tensorflow as tf #для роботи з TensorFlow
from tensorflow import keras #для створення та навчання нейронних мереж
from tensorflow.keras import layers #створення шарів нейронної мережі
from sklearn.preprocessing import LabelEncoder #для кодування міток
import matplotlib.pyplot as plt #для візуалізації даних, побудова графіків

#робота з csv файлом
# використовуємо розширений датасет з папки Data
df = pd.read_csv('Data/figures_training_dataset.csv')

endcoder = LabelEncoder() #ініціалізація об'єкта кодувальника міток
df['label_enc'] = endcoder.fit_transform(df['label']) #кодування міток у числовий формат

# додаємо нову ознаку - співвідношення площі до периметра
df['area_to_perimeter_ratio'] = df['area'] / df['perimeter']

X = df[['area', 'perimeter', 'corners', 'area_to_perimeter_ratio']] #ознаки
y = df['label_enc']

#створення моделі
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(4,)), #вхідний шар з 8 нейронами (4 ознаки)
    layers.Dense(8, activation='relu'), #прихований шар з 8 нейронами
    layers.Dense(len(df['label_enc'].unique()), activation='softmax') #вихідний шар з кількістю нейронів, що відповідає кількості класів
])

#навчання моделі
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X, y, epochs=500, verbose = 0)

#візуалізація процесу навчання
plt.plot(history.history['loss'], label = 'Loss')
plt.plot(history.history['accuracy'], label = 'Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Процес навчання моделі')
plt.legend()
plt.show()

# тестування моделі на нових даних
# додаємо приблизні ознаки для кожного класу, що містяться в датасеті
# формат: [area, perimeter, corners, area_to_perimeter_ratio]
test_data = np.array([
    [30, 25, 3, 30/25],     # triangle (приблизно)
    [100, 40, 4, 100/40],   # square (приблизно)
    [250, 60, 0, 250/60],   # circle (приблизно)
    [110, 40, 5, 110/40],   # pentagon (приблизно)
    [200, 60, 6, 200/60],   # hexagon (приблизно)
    [120, 100, 10, 120/100] # star (приблизно)
])

pred = model.predict(test_data)

print('Імовірності по класах для кожного тестового прикладу:')
for i, probs in enumerate(pred):
    pred_label = endcoder.inverse_transform([np.argmax(probs)])[0]
    print(f'Приклад {i+1} (ознаки={test_data[i].tolist()}):')
    print(f'  Передбачений клас: {pred_label}')
    print(f'  Вектор ймовірностей (отримано): {np.round(probs,4)}')
