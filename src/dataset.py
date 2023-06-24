from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import pandas as pd
import numpy as np
import os

os.chdir("..")

class GermanEmotionDataset:

    def __init__(self, max_length) -> None:
        self.DF_TEST, self.DF_TRAIN = self.import_test_train(False)
        self.LABELS = self.DF_TEST['label_id'].values
        self.MAX_LENGTH = max_length
        

    def import_test_train(self, local):
        """
        This imports the given train and testset locally or not and returns it.

        :param local: If set to true, it will return the trainset from a local view. Otherwise it will open drive mount and attempts to connect to your
        drive folders.
        """
        assert type(local) == bool, f"Type is not valid. Expected boolean, recieved: {type(local)}"

        if local:
            from google.colab import drive
            drive.mount('/content/gdrive')

            df_test = pd.read_csv('/content/gdrive/MyDrive/Experiment/testset_DE_Trigger.csv')
            df_train = pd.read_csv('/content/gdrive/MyDrive/Experiment/trainset_DE_Trigger.csv')

            return df_test, df_train

        else:
            df_test = pd.read_csv('./Experiment/testset_DE_Trigger.csv')
            df_train = pd.read_csv('./Experiment/trainset_DE_Trigger.csv')

            return df_test, df_train


    def tokenize_data(self, tokenizer):
        """Prepares the dataset by predefining the shape of the input (masing, ids, labels). 
        Furthermore tokenizes the train and testset based on a BERT Tokenizer.
        """
        #creating mask for tokens
        self.Xids_train=np.zeros((self.DF_TRAIN.shape[0],self.MAX_LENGTH))
        self.Xmask_train=np.zeros((self.DF_TRAIN.shape[0],self.MAX_LENGTH))
        self.y_train=np.zeros((self.DF_TRAIN.shape[0],1))

        #creating mask for tokens
        self.Xids_test=np.zeros((self.DF_TEST.shape[0],self.MAX_LENGTH))
        self.Xmask_test=np.zeros((self.DF_TEST.shape[0],self.MAX_LENGTH))
        
        # Tokenizing trainset
        for i,sequence in enumerate(self.DF_TRAIN['content']):
            tokens=tokenizer.encode_plus(sequence,max_length=self.MAX_LENGTH,padding='max_length',add_special_tokens=True,
                                truncation=True,return_token_type_ids=False,return_attention_mask=True,
                                return_tensors='tf')

            self.Xids_train[i,:] = tokens['input_ids']
            self.Xmask_train[i,:] = tokens['attention_mask']
            self.y_train[i,0] = self.DF_TRAIN.loc[i,'label_id']

        self.y_train = to_categorical(self.y_train)

        # Tokenizing trainset
        for i,sequence in enumerate(self.DF_TEST['content']):
            tokens=tokenizer.encode_plus(sequence,max_length=self.MAX_LENGTH,padding='max_length',add_special_tokens=True,
                                truncation=True,return_token_type_ids=False,return_attention_mask=True,
                                return_tensors='tf')

            self.Xids_test[i,:] = tokens['input_ids']
            self.Xmask_test[i,:] = tokens['attention_mask']


    def prepare_resulting_test_trainset(self):
        """Prepares the resulting train and testset. The resulting data is compatible 
        for the training.

        :return train, val: returns a train and validation set in the compatible format for training purposes.
        """
        def map_func(input_ids,mask,labels):
            return {'input_ids':input_ids,'attention_mask':mask},labels
        def map_func_test(input_ids,mask):
            return {'input_ids':input_ids,'attention_mask':mask}
        
        dataset=tf.data.Dataset.from_tensor_slices((self.Xids_train,self.Xmask_train,self.y_train))
        dataset=dataset.map(map_func)
        dataset=dataset.shuffle(100000).batch(64).prefetch(1000)

        DS_size=len(list(dataset))

        train=dataset.take(round(DS_size*0.90))
        val=dataset.skip(round(DS_size*0.90))

        dataset_test=tf.data.Dataset.from_tensor_slices((self.Xids_test,self.Xmask_test))
        dataset_test=dataset_test.map(map_func_test)
        # batching it to or the predictions will be multiplied by the shape
        dataset_test=dataset_test.batch(64).prefetch(1000)

        return train, val, dataset_test