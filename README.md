# TrainExportToONNX

This project demonstrates a text classification pipeline using BERT, PyTorch, and ONNX. It includes data preprocessing, model training, evaluation, and exporting the trained model to ONNX format.

## Features
- Text classification using BERT
- Data preprocessing and cleaning
- Model training and evaluation
- Exporting the trained model to ONNX format

## Prerequisites
- Python 3.9 or later
- Install dependencies using the provided `requirements.txt` file:

```bash
py -m pip install -r requirements.txt
```

## Files
- `sample_script.py`: Main script for training and evaluating the model.
- `create_json.py`: Script to generate a sample JSON dataset (`L1Files.json`).
- `L1Files.json`: Sample dataset for training and testing.
- `requirements.txt`: List of required Python packages.

## Usage

1. **Generate the Sample Dataset**:
   Run the `create_json.py` script to create the `L1Files.json` file:
   ```bash
   py create_json.py
   ```

2. **Train and Evaluate the Model**:
   Run the `sample_script.py` script:
   ```bash
   py sample_script.py
   ```

   Follow the prompts to load a saved model or train a new one. The script will output the model's accuracy, classification report, and confusion matrix.

3. **Export the Model to ONNX**:
   If you choose to save the trained model, it will also be exported to ONNX format as `bert_sequence_classification.onnx`.

## Notes
- The sample dataset (`L1Files.json`) is small and intended for demonstration purposes. Replace it with a larger, real-world dataset for better results.
- Ensure that the `transformers` and `torch` libraries are compatible with your Python version.

## License
This project is licensed under the MIT License.
