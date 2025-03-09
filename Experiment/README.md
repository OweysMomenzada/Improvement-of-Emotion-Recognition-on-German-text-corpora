# Experiments

Here, you can run the experiments and reproduce the results shown in the paper.

### Setup
Please install Jupyter Notebook based on [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html
"Anaconda"). Alternatively, you can use free GPU environments like Google Colab without the need for additional setup. Most of the requirements are installed in Google Colab (if additional requirements are needed, they are defined in the notebooks).

If you want to work on your local machine, run the following requirements:
```bash
pip install -r requirements.txt
```

If you work locally, you can ignore the `pip install` commands integrated into the notebooks.

### Data overview
Here is a short description and overview of the datasets:
- `testset_DE_Trigger.csv`: Testset of the German final emotion dataset with the data based on Weak Supervision.
- `trainset_DE_Trigger.csv`: Trainset of the German final emotion dataset with the data based on Weak Supervision.
- `translated_fullset.csv`: Full dataset for the German and English emotion dataset **without** data based on Weak Supervision.

### Experiment overview
Each experiment has a reference to the relevant section of the paper. However, here is a short description and overview of the directories and experiments:

- `ML Models`: Only has one notebook which includes the training and evaluation of all traditional Machine Learning models used in the paper.
- `DL models (with TL)`: This directory includes the training and evaluation of all Deep Learning models (additionally with Transfer Learning) used in the paper.
- `Transformer Models`: This directory includes all experiments which utilized the BERT model.
    * `German BERT.ipynb`: Notebook which evaluates and trains the German BERT model based on the German final emotion dataset with the data based on Weak Supervision.
    * `Multilingual BERT`:  Notebook which evaluates and trains the Multilingual BERT model based on the German final emotion dataset with the data based on Weak Supervision.
    * `Subset training`: All training and evaluations of the BERT (German as well as English) model based on each subset (domain) instead of using Multiview learning. Meaning that this folder only includes training without combining the all domains. Furthermore, the performance of the English and German BERT as well as the accuracy of the NMT approach are compared.
    * `Domain generalization`: This directory includes the experiment for domain generalization, meaning how the model performs on training on the whole trainset excluding the two least similar subsets and then testing the trained model on those two subsets.
    * `Sentiment Analysis`: This experiment transforms the ER System into a Sentiment Analysis approach and compares it to existing Sentiment Analysis approaches for the German language. **Note!** Here you need the finetuned model resulting from the `src` folder.
- `LLM Approach`: A short experiment on how the model could be utilized for chatbots using Large Language models.