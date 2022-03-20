import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
from tqdm import tqdm
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
#import tensorflow_hub as hub
# os.environ["TFHUB_DOWNLOAD_PROGRESS"]="1"
from datset import get_loaders

URL = "https://tfhub.dev/google/efficientnet/b3/feature-vector/1"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
#MODEL_PATH = os.path.join("efficientb0")
LOAD_MODEL = False
NUM_CLASSES = 70

def get_model(img_size,num_classes):
    base_network = MobileNetV2(input_shape=(img_size,img_size,3),include_top=False,pooling='avg')
    base_network.trainable = False
    inputs = base_network.input
    x = tf.keras.layers.Dense(1000,activation='relu')(base_network.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(units=num_classes,activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs,outputs=outputs)
    return model

@tf.function
def train_step(data,labels,acc_metric,model,loss_fn,optimizer):
    with tf.GradientTape() as tape:
        predictions = model(data,training=True)
        loss = loss_fn(labels,predictions)

    gradients = tape.gradient(loss,model.trainable_weights)
    optimizer.apply_gradients(zip(gradients,model.trainable_weights))
    acc_metric.update_state(labels,predictions)
    return loss,acc_metric.result()

def evaluate_model(ds_validation,model):
    accuracy_metric = keras.metrics.SparseCategoricalAccuracy()
    for idx , (data,labels) in enumerate(ds_validation):
        data = data.permute(0,2,3,1)
        data = tf.convert_to_tensor(np.array(data))
        labels = tf.convert_to_tensor(np.array(labels))
        y_pred = model(data,training=False)
        accuracy_metric.update_state(labels,y_pred)

    accuracy = accuracy_metric.result()
    print(f'Accuracy over validation set : {accuracy}')


if __name__ == "__main__" :
    physical_devices = tf.config.list_physical_devices("GPU")
    try :
        tf.config.experimental.set_memory_growth(physical_devices[0],True)
        print("you 're using GPU'")
    except:
        print("you are using CPU")
    model = get_model(IMG_SIZE,NUM_CLASSES)
    #model.summary()
    test_path = os.path.join("data","test")
    model_path = os.path.join("model.h5")
    train_path = os.path.join("data","train")
    valid_path = os.path.join("data","valid")
    train_loader , val_loader = get_loaders(train_path,valid_path,BATCH_SIZE,IMG_SIZE)
    optimizer = keras.optimizers.Adam(lr=0.0001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    acc_metric = keras.metrics.SparseCategoricalAccuracy()

    #Training loop
    for epoch in range(NUM_EPOCHS):
        for idx,(data,labels) in enumerate(tqdm(train_loader)):
            data = data.permute(0,2,3,1)
            data = tf.convert_to_tensor(np.array(data))
            labels = tf.convert_to_tensor(np.array(labels))
            los,acc=train_step(data,labels,acc_metric,model,loss_fn,optimizer)

            if idx % 10 == 0 and idx > 0 :
                #train_acc = acc_metric.result()
                #loss = loss.numpy()
                print(f"Accuracy over epoch {acc}")
                print(f"Loss over epoch {los}")

                evaluate_model(val_loader,model)
                model.save(model_path)

