import pandas as pd #для роботи з даними у форматі таблиць
import numpy as np #для роботи з масивами та таблицями
import tensorflow as tf #для роботи з TensorFlow
from tensorflow import keras #для створення та навчання нейронних мереж
from tensorflow.keras import layers #створення шарів нейронної мережі
from sklearn.preprocessing import LabelEncoder #для кодування міток
import matplotlib.pyplot as plt #для візуалізації даних, побудова графіків

#робота з csv файлом
df = pd.read_csv('figures.csv')

endcoder = LabelEncoder() #ініціалізація об'єкта кодувальника міток
df['label_enc'] = endcoder.fit_transform(df['label']) #кодування міток у числовий формат

X = df[['area', 'perimeter', 'corners']] #ознаки
y = df['label_enc']

#створення моделі
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(3,)), #вхідний шар з 8 нейронами
    layers.Dense(8, activation='relu'), #прихований шар з 8 нейронами
    layers.Dense(len(df['label_enc'].unique()), activation='softmax') #вихідний шар з кількістю нейронів, що відповідає кількості класів
])

#навчання моделі
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X, y, epochs=200, verbose = 0)

#візуалізація процесу навчання
plt.plot(history.history['loss'], label = 'Loss')
plt.plot(history.history['accuracy'], label = 'Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Процес навчання моделі')
plt.legend()
plt.show()

#тестування моделі на нових даних
test_data = np.array([[15, 14, 4],  #приблизні ознаки для квадрата
                      [12, 13, 3],  #приблизні ознаки для трикутника
                      [18, 16, 0]]) #приблизні ознаки для кола
pred = model.predict(test_data)
print(f'Імовірність по кожному класу: {pred}')
print(f'Модель визначила: {endcoder.inverse_transform([np.argmax(pred)])}')