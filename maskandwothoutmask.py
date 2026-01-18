import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
#from google.colab.patches import cv2_imshow
from PIL import Image 
from sklearn.model_selection import train_test_split

with_mask_files = os.listdir('C:\Data D\Deep Lerning Projects\data\with_mask')
print(with_mask_files[0:5])
print(with_mask_files[-5:],'\n')

without_mask_files = os.listdir('C:\Data D\Deep Lerning Projects\data\without_mask')
print(without_mask_files[0:5])
print(without_mask_files[-5:])

print('Number Of With Mask Image:',len(with_mask_files))
print('Number Of With Out Mask Image:',len(without_mask_files))

#create the label 0 and 1 with mask 1 and without mask 0 Means Label Encoding
with_mask_label = [1]*3725
with_maskout_label = [0]*3828

print(with_mask_label[0:5])
print(with_maskout_label[0:5])

#add to list
# labels = with_mask_label + with_maskout_label
# print(labels[0:5])
# print(labels[-5:])

#display images mask images
# img = mpimg.imread('C:\Data D\Deep Lerning Projects\data\with_mask\with_mask_1545.jpg')
# imgplot = plt.imshow(img)
# plt.show()

#display image without masks
# img  = mpimg.imread('C:\Data D\Deep Lerning Projects\data/without_mask/without_mask_2925.jpg')
# imgplot = plt.imshow(img)
# plt.show()

#image Proccesing numpy to array+
#1,resize the images
#2,convert image in numpy into array


with_mask_path = 'C:/Data D/Deep Lerning Projects/data/with_mask/'

data = []
labels = []
for img_file in os.listdir(with_mask_path):  
      img_path = os.path.join(with_mask_path, img_file)
      try:
          image = Image.open(img_path)
          image = image.resize((128, 128))
          image = image.convert('RGB')
          image = np.array(image)
          data.append(image)
          labels.append(1)  # With mask = 1
      except Exception as e:
          print(f"Error loading {img_path}: {e}")

without_mask_path = 'C:/Data D/Deep Lerning Projects/data/without_mask/'

for img_file in os.listdir(without_mask_path):  
        img_path = os.path.join(without_mask_path, img_file)
        try:
            image = Image.open(img_path)
            image = image.resize((128, 128))
            image = image.convert('RGB')
            image = np.array(image)
            data.append(image)
            labels.append(0)  # Without mask = 0
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

#print(len(data))

#converting image list and label into numpy array
x = np.array(data)
y = np.array(labels)

# print(type(x))
# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
#print(x.shape,x_train,shape,x_test.shape)
#scaling the data

x_train_scaled = x_train / 255.0
x_test_scaled = x_test / 225.0

#print(x_train_scaled)

#builde the cnn convelusanal nural network actual model create
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Flatten,Dropout

#how predict like first was mask and seconde was without mask 
model = keras.Sequential()

model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation = 'relu',input_shape=(128,128,3)))#3 means rgb
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(1,activation='sigmoid'))

#compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,loss = 'binary_crossentropy',metrics = ['accuracy'])
#training a nural network
history = model.fit(x_train_scaled,y_train,validation_split=0.1,epochs=4,batch_size=32,verbose=1)

loss,accuracy = model.evaluate(x_test_scaled,y_test)
print(accuracy * 100)

h = history

#plot the loss value
plt.plot(h.history['loss'],label='Train Loss')
plt.plot(h.history['val_loss'],label='Validation Loss')
plt.legend()
plt.show()

plt.plot(h.history['accuracy'],label = 'Train Accuracy')
plt.plot(h.history['val_accuracy'],label = 'Validation Accuracy')
plt.legend()
plt.show()

#Predictive System
input_image_path = input("Enter Image Path: ")

input_image = cv2.imread(input_image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (128,128))
input_image = input_image / 255.0
input_image = np.reshape(input_image, (1,128,128,3))

prediction = model.predict(input_image)

if prediction[0][0] > 0.5:
    print("✅ The person is WEARING a mask")
else:
    print("❌ The person is NOT wearing a mask")

input_image_path = input("Enter Image Path: ")

input_image = cv2.imread(input_image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (128,128))
input_image = input_image / 255.0
input_image = np.reshape(input_image, (1,128,128,3))

prediction = model.predict(input_image)

if prediction[0][0] > 0.5:
    print("✅ The person is WEARING a mask")
else:
    print("❌ The person is NOT wearing a mask")
    
model.save("mask_detector_model.h5")

# input_image_path = input('Path Of The Image To Be Predicted:')
# input_image = cv2.imread(input_image_path)
# # cv2.imshow(input_image)
# input_image_resized = cv2.resize(input_image, (128,128))
# input_image_scaled = input_image_resized/255
# input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])
# input_prediction = model.predict(input_image_reshaped)
# print(input_prediction)
# input_pred_label = np.argmax(input_prediction)
# print(input_pred_label)


# if input_pred_label == 1:
#   print('The person in the image is wearing a mask')
# else:
#   print('The person in the image is not wearing a mask')

# input_image_path = input('Path Of The Image To Be Predicted:')
# input_image = cv2.imread(input_image_path)
# # cv2.imshow(input_image)
# input_image_resized = cv2.resize(input_image, (128,128))
# input_image_scaled = input_image_resized/255
# input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])
# input_prediction = model.predict(input_image_reshaped)
# print(input_prediction)
# input_pred_label = np.argmax(input_prediction)
# print(input_pred_label)


# if input_pred_label == 1:
#   print('The person in the image is wearing a mask')
# else:
#   print('The person in the image is not wearing a mask')



