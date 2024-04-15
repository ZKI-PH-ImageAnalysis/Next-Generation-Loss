import tensorflow

from matplotlib import pyplot

from sklearn.model_selection  import train_test_split

from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout, UpSampling2D, BatchNormalization


class NGL(tensorflow.keras.losses.Loss):
    def __init__(
    	self, 
    	scaling=False,
    	name="ngl_loss"):
        super().__init__(name=name)
        self.name = name
        self.scaling = scaling

    def call(self, y_true, y_pred):
        y_true = tensorflow.cast(y_true, tensorflow.float32)
        y_pred = tensorflow.cast(y_pred, tensorflow.float32)
        if self.scaling == True:
	 	        y_pred = tensorflow.math.sigmoid(y_pred)
        part_1 = tensorflow.math.exp(2.4092 - y_pred - y_pred*y_true)
        part_2 = tensorflow.math.cos(tensorflow.math.cos(tensorflow.math.sin(y_pred)))
        elements = part_1 - part_2
        loss = tensorflow.reduce_mean(elements)
        return loss


def preprocessed_dataset():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.astype('float32') 
    x_test = x_test.astype('float32') 
    
    y_train = to_categorical(y_train, num_classes=100)
    y_test = to_categorical(y_test, num_classes=100)
    
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)
    
    x_train, x_val, y_train, y_val = train_test_split(np.asarray(x_train), 
                                                      np.asarray(y_train), 
                                                      test_size=0.1, 
                                                      random_state = 42
                                                      )
    
    train_DataGen = ImageDataGenerator(zoom_range=0.2, 
                                       width_shift_range=0.1, 
                                       height_shift_range = 0.1, 
                                       horizontal_flip=True
                                       )
     
    valid_datagen = ImageDataGenerator()
    
    test_datagen = ImageDataGenerator()
    
    train_set = train_DataGen.flow(x_train, y_train, batch_size=128)
    valid_set = valid_datagen.flow(x_val, y_val, batch_size=128) 
    test_set = test_datagen.flow(x_test, y_test, batch_size=1) 
    
    return train_set, valid_set, test_set


train_set, valid_set, test_set = preprocessed_dataset()
        
model = None
model = Sequential()
model.add(UpSampling2D())
model.add(UpSampling2D())
model.add(UpSampling2D())

inc_model = InceptionV3(include_top = False, weights = None, pooling = 'max', classes = 100)
for layer in inc_model.layers:
    layer.trainable = True
            
model.add(inc_model)
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(100, activation = 'softmax'))
        
model.compile(optimizer='adam', loss=NGL(scaling=True), metrics=['accuracy'])
        
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-4)
        
model.build(input_shape = (None, 32, 32, 3))
model.summary()

filename = 'supplementary.csv'
csv_logger = CSVLogger(filename)

model.fit(train_set, 
          batch_size=128, 
          validation_data = valid_set,
          epochs = 200, 
          callbacks=[reduce_lr, csv_logger],
          verbose = 1
          )
    
loss, acc = model.evaluate(test_set, verbose=0)

f = open(filename, "a")
f.write(str(acc))
f.write('\n')
f.close()
