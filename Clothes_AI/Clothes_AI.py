import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

data = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = data.load_data()

modified_test = test_images.copy()

train_images = train_images/255
test_images = test_images/255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation = "relu"),
    keras.layers.Dense(10,activation = "softmax")
    ])



model.compile(optimizer="adadelta",loss = "sparse_categorical_crossentropy",metrics = ["accuracy"])
model.fit(train_images,train_labels,epochs = 5)


test = modified_test.copy()
#Codigo para adicionar ruidos simples nas imagens.
"""
for v in test:
  for i in range(10,15):
    for j in range(0,8):
      v[i][j] = 255
"""
#-------------------------------------------
plt.grid(0)
plt.axis("off")
plt.imshow(test[10],cmap = plt.cm.binary)
plt.show

test = test/255

test_loss,test_acc = model.evaluate(test,test_labels)
print("Tested Acc:",test_acc)
