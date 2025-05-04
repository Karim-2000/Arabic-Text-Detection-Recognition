# ComputerVisionProject


Arabic Text Detection and Recognition
This repository implements an AI system for detecting and recognizing Arabic text in natural scene images using a modified EAST detector with attention and a transformer-based recognizer, optimized for a T4 GPU.
Dataset
We used the EvArEST dataset, which includes training and testing splits for Arabic and English scene text. Download from https://github.com/HGamal11/EvArEST-dataset-for-Arabic-scene-text.

Training: Used for model training (/content/trainingdataimages/ or EvArEST/train/).
Testing: Used for evaluation (/content/testingdataimages/ or EvArEST/test/).
Note: The code processes only Arabic text annotations, filtering out English text to focus on Arabic script challenges.

Setup

Install dependencies (preferably in a Google Colab environment):pip install torch torchvision opencv-python pillow numpy python-Levenshtein


Download the EvArEST dataset:
Place it in /content/ with subfolders trainingdataimages/, trainingdataannotations/, testingdataimages/, and testingdataannotations/.
Alternatively, use EvArEST/train/ and EvArEST/test/ and update paths in the script.


Mount Google Drive (if using Colab):from google.colab import drive
drive.mount('/content/drive')

Update paths in the script if storing data in Google Drive (e.g., /content/drive/MyDrive/EvArEST/).
Save the script:
Copy the code into a .py file (e.g., main.py) or run it in a Colab notebook.



Testing a Single Image
To verify dataset loading for one image (e.g., img_1.jpg):
python main.py

The script loads the first image from the training set, prints its bounding boxes and Arabic texts, and displays/saves the image with bounding boxes as output.jpg (set display=True in test_single_image). Ensure dataset paths are correct.
Training
Run the training script:
python main.py

The model trains on a T4 GPU (or CPU if no GPU is available) using the training split (Arabic text only) for 5 epochs by default. Model weights are saved as detector.pth and recognizer.pth.
Testing
To evaluate the model on the testing split:

Train the model or load trained weights:detector.load_state_dict(torch.load("detector.pth"))
recognizer.load_state_dict(torch.load("recognizer.pth"))


Run evaluation:evaluate_model(detector, recognizer, test_dataloader, device, vocab)



The evaluation outputs a mock F1-score (0.85) for detection and Character Error Rate (CER) for recognition.
Requirements

Python 3.8+
PyTorch 1.9+
CUDA-enabled T4 GPU (14.74 GiB) for training
EvArEST dataset with images and annotations
python-Levenshtein for CER calculation (install with pip install python-Levenshtein)

Implementation Details

Dataset Handling:
Annotations are in the format: x1,y1,x2,y2,x3,y3,x4,y4,language,text. The code filters for language=Arabic and uses the 10th field as text.
Bounding box coordinates are stored in the original image scale and scaled to 64x64 for processing (32x32 for ROI cropping in recognition).


Models:
EASTWithAttention: Uses MobileNetV2 backbone with an attention mechanism to enhance detection of complex Arabic calligraphy.
TransformerRecognizer: Simplified transformer with d_model=64, nhead=2, num_layers=1, and 32x32 input resolution (sequence_length=256).


Optimizations:
Batch Size: Set to 1 to minimize memory usage.
Gradient Accumulation: Uses accum_steps=2 for an effective batch size of 2.
Automatic Mixed Precision (AMP): Reduces memory usage via torch.cuda.amp.
Memory Management:
Sequential processing of detector and recognizer with torch.no_grad() for cropping.
Aggressive memory cleanup using torch.cuda.empty_cache() and gc.collect() after each forward pass and batch.
Stored detector loss value to avoid accessing deleted tensors.
ROI cropping resized to 32x32 to further reduce memory demands.


ROI Processing: Recognition processes cropped regions based on bounding boxes, ensuring alignment with text annotations.



Notes

Adjust vocab_size based on the Arabic character set in the dataset.
The single-image test validates dataset loading, with correct bounding box visualization on original images.
If python-Levenshtein is unavailable, a fallback Levenshtein function is used (slower, less optimized).
A custom collate function handles variable-length annotations in the DataLoader.
The script skips batches with no valid texts or boxes to prevent errors.
Environment variable PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set to avoid memory fragmentation.

Known Limitations

The mock F1-score (0.85) is used for detection evaluation; implement IoU-based metrics for accurate results.
The recognizer uses only the first output of the transformer sequence, which may limit accuracy for longer texts.
Training on 32x32 ROIs may reduce recognition accuracy but is necessary for T4 GPU compatibility.

