#!/usr/bin/env python3

from keras.applications.inception_v3 import InceptionV3
from keras import preprocessing
from keras import backend
from keras.layers import Dense, GlobalAveragePooling2D
from keras import initializers
from keras.models import Model

train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory="/projects/bgmp/oda/training/sliced/110_blank_label/",
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=256,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)

backend.set_image_data_format("channels_last")

base_model = InceptionV3(weights='imagenet', include_top=False,
                         input_shape=(256, 256, 3))

out = base_model.output

out = GlobalAveragePooling2D()(out)

out = Dense(1024, activation='relu',
            kernel_initializer=initializers.VarianceScaling(scale=2.0))(out)

predictions = Dense(2, activation='sigmoid',
                    kernel_initializer=initializers.VarianceScaling(scale=2.0))(out)

model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=133, epochs=20)
