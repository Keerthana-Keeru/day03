from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.optimizers import Adam
from keras.datasets import fashion_mnist,cifar10
from keras import regularizers
import matplotlib.pyplot as plt
from keras.utils import to_categorical
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#normalize
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

# to categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#architecture
model_base=Sequential()
model_base.add(Flatten(input_shape=(32,32,3)))
model_base.add(Dense(1024,activation='relu'))
model_base.add(Dense(512,activation='relu'))
model_base.add(Dense(256,activation='relu'))
model_base.add(Dense(128,activation='relu'))
model_base.add(Dense(64,activation='relu'))
model_base.add(Dense(10,activation='softmax'))

#compile
model_base.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
#train
history=model_base.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=(0.2))

#evaluate
loss,test_accuracy=model_base.evaluate(x_test,y_test)
print(f"test accuracy 1:{test_accuracy}")


#________________________________________________________________________________________________________________________________
# Model 2 with L2 regularizer(le-4) and dropout

#architecture
model_le4=Sequential()
model_le4.add(Flatten(input_shape=(32,32,3)))
model_le4.add(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(1e-4)))
model_le4.add(Dense(512,activation='relu',kernel_regularizer=regularizers.l2(1e-4)))
model_le4.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l2(1e-4)))
model_le4.add(Dense(128,activation='relu',kernel_regularizer=regularizers.l2(1e-4)))
model_le4.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(1e-4)))
model_le4.add(Dense(10,activation='softmax'))

#compile
model_le4.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
#train
history_model_2=model_le4.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=(0.2))

#evaluate
loss,test_accuracy=model_le4.evaluate(x_test,y_test)
print(f"test accuracy:2 {test_accuracy}")

#_____________________________________________________________________________________________________________________________
# Model 3 with L2 regularizer(le-4) and dropout

#architecture
model_le2=Sequential()
model_le2.add(Flatten(input_shape=(32,32,3)))
model_le2.add(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(1e-2)))
model_le2.add(Dense(512,activation='relu',kernel_regularizer=regularizers.l2(1e-2)))
model_le2.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l2(1e-2)))
model_le2.add(Dense(128,activation='relu',kernel_regularizer=regularizers.l2(1e-2)))
model_le2.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(1e-2)))
model_le2.add(Dense(10,activation='softmax'))

#compile
model_le2.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
#train
history_model_3=model_le2.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=(0.2))

#evaluate
loss,test_accuracy=model_le2.evaluate(x_test,y_test)
print(f"test accuracy:3{test_accuracy}")
#_________________________________________________________________________________________________________________________
# visualization
plt.plot(history.history['val_accuracy'],label='without regularizer',color='red')
plt.plot(history_model_2.history['val_accuracy'],label='le-4-model',color='blue')
plt.plot(history_model_3.history['val_accuracy'],label='le-2-model',color='green')
plt.title("validatio accuracy")
plt.xlabel("epochs")
plt.ylabel("accuarcy")
plt.legend()
plt.show()