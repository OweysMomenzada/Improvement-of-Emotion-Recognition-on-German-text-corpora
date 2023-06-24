# Implementation of the ER System

Here, you can run the ER system on your own machine. The chosen parameters and setups were based on the investigations conducted (see `./Experiment/` folder).

### Setup
To get started, please install Python 3.8 or higher. Next, create a new Python environment and install all dependencies by running the following command:
```bash
pip install -r requirements.txt
```
This will install all the necessary dependencies for the ER system.

To train the BERT model on the relevant data, execute the train.py script. It is highly recommended to train the model on a GPU for faster performance. If using a GPU, make sure to follow the [GPU installation](https://wandb.ai/wandb/common-ml-errors/reports/How-To-Install-TensorFlow-With-GPU-Support-on-Windows--VmlldzozMDYxMDQ "installation guide") to set up the required GPU support. Alternatively, you can use free GPU environments like Google Colab without the need for additional setup.
```bash
python train.py
```

After successful training, you can use the ER system as shown in the following example (executed in a Jupyter Notebook located in the src directory):
```python
from src import model
from src.utils import config

er_model = model.GermanEmotionBERT()
er_model.load_model(checkpoint_path=config.CHECKPOINT_PATH)
print(er_model.predict("Heute ist ein fantastischer Tag, um mehr über Transformers zu erfahren!"))

>>> {'anger': 0.0005,
>>> 'fear': 0.0009,
>>> 'joy': 0.9518,
>>> 'neutral': 0.0454,
>>> 'sadness': 0.0014}
```

Please note that the above example assumes the usage of a Jupyter Notebook and importing the necessary modules from the src directory. Adjust the code as needed based on your specific implementation.