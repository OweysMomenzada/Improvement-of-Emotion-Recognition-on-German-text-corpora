# Data Collecting

Here, a simplified implementation of the Neural Machine Translation (NMT) and Weak Supervision approach is shown. 
For Weak Supervision, the DWDS API was used based on the `triggerwords.xlsx`, which is a German Emotion Lexicon. The `emotiondataset_builder` combines both NMT and the Weak Supervision approach. **Note**, that the NMT used here is the free GoogleTranslator from `deep_translator`. To translate the final dataset based on [DeepL](https://www.deepl.com/de/docs-api
"DeepL") you need to set up a billing plan and create a token to be able to translate the whole data.

### Overview
Here an overview of the `emotiondataset_builder` is provided.

- `generate_example_text_based_on_keywords`: The function provides the Weak Supervision approach.
- `translate_texts`: Translates the text with `deep_translator`

### Example usage
For both functions, examples are shown after executing `emotiondataset_builder`:

Install the requirements first:
```bash
pip install -r requirements.txt
```

Run the examples: 
```bash
python emotiondataset_builder.py
```