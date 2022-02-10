from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from SimplePreprocessor import SimplePreprocessor
from DataLoader import DataLoader
from ShallowNet import ShallowNet
from tensorflow.keras.optimizers import SGD, Adam
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from vgg import VGG
from datetime import datetime

dataset = "path/to/dataset/"
n_epochs = 500

print("Loading Images")
imagePaths = list(paths.list_images(dataset))

print(imagePaths)
sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor(dataFormat="channels_last")

sdl = DataLoader(preprocessors=[sp,iap])
data, txt_labels = sdl.load(imagePaths, verbose = 20)
data = data.astype("float") / 255.0

lb = LabelBinarizer()
labels = lb.fit_transform(txt_labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# initialize the optimizer and model
start = datetime.now()
print("[INFO] compiling model...")
opt = SGD(learning_rate=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=len(lb.classes_))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=8, epochs=500, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=8)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
end = datetime.now()
print("ShallowNet took: ", str(end-start), "to complete")
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,n_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,n_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,n_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,n_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# TRAINING VGG

start = datetime.now()
print("[INFO] compiling model...")
opt = SGD(learning_rate=0.005)
model = VGG.build(width=32, height=32, depth=3, classes=len(lb.classes_))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=8, epochs=500, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=8)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
end = datetime.now()
print("VGG-16 took: ", str(end-start), "to complete")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,n_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,n_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,n_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,n_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
