import pandas as pd
import numpy as np
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Dense, BatchNormalization, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2
from sklearn.metrics import confusion_matrix

df = pd.read_csv('fer2013.csv')
df.head()
X_train = []
y_train = []
X_test = []
y_test = []
for index, row in df.iterrows():
    k = row['pixels'].split(" ")
    if row['Usage'] == 'Training':
        X_train.append(np.array(k))
        y_train.append(row['emotion'])
    elif row['Usage'] == 'PublicTest':
        X_test.append(np.array(k))
        y_test.append(row['emotion'])
X_train = np.array(X_train, dtype = 'uint8')
y_train = np.array(y_train, dtype = 'uint8')
X_test = np.array(X_test, dtype = 'uint8')
y_test = np.array(y_test, dtype = 'uint8')

y_train= to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
 
datagen = ImageDataGenerator( 
    rescale=1./255,
    rotation_range = 10,
    width_shift_range=0.1,
    horizontal_flip = True,
    height_shift_range=0.1,
    fill_mode = 'nearest')

testgen = ImageDataGenerator( 
    rescale=1./255
    )
datagen.fit(X_train)

batch_size = 64
train_flow = datagen.flow(X_train, y_train, batch_size=batch_size) 
test_flow = testgen.flow(X_test, y_test, batch_size=batch_size)
def FER_Model(input_shape=(48,48,1)):
    # first input model
    visible = Input(shape=input_shape, name='input')
    num_classes = 7
    #the a block
    conva_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conva_1')(visible)
    conva_1 = BatchNormalization()(conva_1)
    conva_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conva_2')(conva_1)
    conva_2 = BatchNormalization()(conva_2)
    poola_1 = MaxPooling2D(pool_size=(2,2), name = 'poola_1')(conva_2)
    dropa_1 = Dropout(0.3, name = 'dropa_1')(poola_1)
    #the b block
    convb_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'convb_1')(dropa_1)
    convb_1 = BatchNormalization()(convb_1)
    convb_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'convb_2')(convb_1)
    convb_2 = BatchNormalization()(convb_2)
    convb_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'convb_3')(convb_2)
    convb_2 = BatchNormalization()(convb_3)
    poolb_1 = MaxPooling2D(pool_size=(2,2), name = 'poolb_1')(convb_3)
    dropb_1 = Dropout(0.3, name = 'dropb_1')(poolb_1)
    #the c block
    convc_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'convc_1')(dropb_1)
    convc_1 = BatchNormalization()(convc_1)
    convc_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'convc_2')(convc_1)
    convc_2 = BatchNormalization()(convc_2)
    convc_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'convc_3')(convc_2)
    convc_3 = BatchNormalization()(convc_3)
    convc_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'convc_4')(convc_3)
    convc_4 = BatchNormalization()(convc_4)
    poolc_1 = MaxPooling2D(pool_size=(2,2), name = 'poolc_1')(convc_4)
    dropc_1 = Dropout(0.3, name = 'dropc_1')(poolc_1)
    #the d block
    convd_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'convd_1')(dropc_1)
    convd_1 = BatchNormalization()(convd_1)
    convd_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'convd_2')(convd_1)
    convd_2 = BatchNormalization()(convd_2)
    convd_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'convd_3')(convd_2)
    convd_3 = BatchNormalization()(convd_3)
    convd_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'convd_4')(convd_3)    
    convd_4 = BatchNormalization()(convd_4)
    poold_1 = MaxPooling2D(pool_size=(2,2), name = 'poold_1')(convd_4)
    dropd_1 = Dropout(0.3, name = 'dropd_1')(poold_1)
    
    #the e block
    conve_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conve_1')(dropd_1)
    conve_1 = BatchNormalization()(conve_1)
    conve_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conve_2')(conve_1)
    conve_2 = BatchNormalization()(conve_2)
    conve_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conve_3')(conve_2)
    conve_3 = BatchNormalization()(conve_3)
    conve_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conve_4')(conve_3)
    conve_3 = BatchNormalization()(conve_3)
    poole_1 = MaxPooling2D(pool_size=(2,2), name = 'poole_1')(conve_4)
    drope_1 = Dropout(0.3, name = 'drope_1')(poole_1)
    #Flatten and output
    flatten = Flatten(name = 'flatten')(drope_1)
    ouput = Dense(num_classes, activation='softmax', name = 'output')(flatten)

    # create model 
    model = Model(inputs =visible, outputs = ouput)
    # summary layers
    print(model.summary())
    
    return model
model = FER_Model()
opt = Adam(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
num_epochs = 100  
history = model.fit_generator(train_flow, 
                    steps_per_epoch=len(X_train) / batch_size, 
                    epochs=num_epochs,  
                    verbose=1,  
                    validation_data=test_flow,  
                    validation_steps=len(X_test) / batch_size)
model_json = model.to_json()
with open("model_arch.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_weights.h5")
