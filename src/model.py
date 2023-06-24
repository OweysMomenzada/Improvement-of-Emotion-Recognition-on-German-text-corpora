import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

class GermanEmotionBERT:
    def __init__(self, max_length=100) -> None:
        """Initializes tokenizes and German BERT model for fine-tuning.

        :param max_length: defines the max length of tokens being considered for an input.
        """
        self.MAX_LENGTH = max_length
        self.TOKENIZER = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
        self.BERT = TFAutoModel.from_pretrained("dbmdz/bert-base-german-uncased")


    def emotion_model(self):
        """Initializes the relevant layers to train the German BERT model
        """
        input_ids=tf.keras.layers.Input(shape=(self.MAX_LENGTH,),name='input_ids',dtype='int32')
        input_mask=tf.keras.layers.Input(shape=(self.MAX_LENGTH,),name='attention_mask',dtype='int32')

        embedding=self.BERT(input_ids,attention_mask=input_mask)[0]
        x=tf.keras.layers.GlobalMaxPool1D()(embedding)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.Dense(256,activation='relu')(x)
        x=tf.keras.layers.Dropout(0.2)(x)
        output=tf.keras.layers.Dense(5,activation='softmax')(x)

        model=tf.keras.Model(inputs=[input_ids,input_mask],outputs=output)

        model.layers[2].trainable=False

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                    optimizer='adam',metrics=[tf.keras.metrics.AUC()])

        return model
    

    def save_model(self, model, checkpoint_path):
        """Loading model based on a checkpoint path

        Args:
        :param model: Trained BERT model
        :param checkpoint_path: Checkpoint where the model should be saved.
        """
        model.save_weights(checkpoint_path)
        print("INFO: Model saved successfully.")


    def load_model(self, checkpoint_path):
        """Loading model based on a checkpoint path

        Args:
        :param checkpoint_path: Checkpoint where the model is saved to.
        """
        self.loaded_model = self.emotion_model()
        self.loaded_model.load_weights(checkpoint_path)
        print("INFO: Model was loaded.")


    def predict(self, input):
        seq = self.TOKENIZER.encode_plus(input,max_length=self.MAX_LENGTH,padding='max_length',add_special_tokens=True,
                           truncation=True,return_token_type_ids=False,return_attention_mask=True,
                           return_tensors='tf')

        seq = [seq['input_ids'], seq['attention_mask']]

        result = self.loaded_model.predict(seq)[0]
        result = [round(i,4) for i in result]

        dict_res = {'anger':result[0],
                    'fear':result[1],
                    'joy':result[2],
                    'neutral':result[3],
                    'sadness':result[4]
        }

        return dict_res