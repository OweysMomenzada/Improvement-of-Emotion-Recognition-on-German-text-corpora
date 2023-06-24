import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import model
import dataset
import utils.gpu_check as gpu_check
import utils.config as config


# maximum length of the input
MAX_LENGTH = 100

def evaluate_model(model, test, y_true):
    y_pred=model.predict(test)
    y_pred_new = np.argmax(y_pred,axis=1)
    print("INFO: Evaluation metrics")
    print(metrics.classification_report(y_true, y_pred_new))


def train_model(model, train, val):
    """Function uses the trainset and validation set to train the model on the
    emotion dataset.

    :param model: Not fine-tuned BERT model
    :param train: trainset
    :param val: validation set
    """
    print(model.summary())

    # define early stopping conditions
    es = EarlyStopping(
        monitor='val_auc',
        patience=2,
        min_delta=0.0010,
        mode='max'
    )
    # define adam optimizer. Note, that for the experiments we used SGD. 
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    # Train model based and monitor loss
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train,
              validation_data=val,
              epochs=1,
              callbacks=[es])
    
    return model


def entrypoint():
    """Entrypoint of the training.
    """

    # define inital pretrained German BERT
    emotion_bert = model.GermanEmotionBERT()
    # import raw data
    dataset_object = dataset.GermanEmotionDataset(MAX_LENGTH)
    # transform raw data into tokenized data
    dataset_object.tokenize_data(tokenizer=emotion_bert.TOKENIZER)
    # get appropriate format for training
    train, val, test = dataset_object.prepare_resulting_test_trainset()

    # train untrained
    untrained_model = emotion_bert.emotion_model()
    trained_model = train_model(model=untrained_model,
                                train=train, 
                                val=val)

    # evaluate model
    evaluate_model(model=trained_model, 
                   test=test, 
                   y_true=dataset_object.LABELS)

    # save model
    emotion_bert.save_model(model=trained_model, 
                            checkpoint_path=config.CHECKPOINT_PATH)

if __name__ == '__main__':
    gpu_check.gpu_check()
    entrypoint()