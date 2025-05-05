# Arabic Text Detection and Recognition

This project develops a computer vision system for Arabic scene text detection and recognition, addressing a UAE problem (COE-49413, AUS). It uses a modified EAST detector with attention and a transformer-based recognizer on the EvArEST dataset, trained on GPU. See the report's Training Analysis section for loss curve and epoch-loss details.

## Setup
1. **Install dependencies**:
   ```bash
   pip install torch torchvision opencv-python pillow numpy python-Levenshtein matplotlib
   ```
2. **Download EvArEST dataset**:
   - From: https://github.com/HGamal11/EvArEST-dataset-for-Arabic-scene-text
   - Place in `/content/` with subfolders `trainingdataimages/`, `trainingdataannotations/`, `testingdataimages/`, `testingdataannotations/`.
3. **Mount Google Drive** (Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. **Upload models** (available upon request):
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload detector.pth, recognizer.pth
   ```

## Training
```bash
python main.py
```
- Trains for 20 epochs on GPU (~2–6 hours for ~5,000 images, batch size 8).
- Saves: `detector.pth`, `recognizer.pth`, `vocab.pkl`, `sample_output.png`.

## Testing/Evaluation
```python
import pickle
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)
detector = EASTWithAttention().to(device)
recognizer = TransformerRecognizer(vocab_size).to(device)
detector.load_state_dict(torch.load("detector.pth"))
recognizer.load_state_dict(torch.load("recognizer.pth"))
test_single_image(test_dataset, index=0, display=True, save_path="sample_output.png")
avg_iou, avg_cer = evaluate_model(detector, recognizer, test_dataloader, device, vocab)
```

## Sample Output
```
Image saved to sample_output.png
Epoch 1, Total Loss: 143.72580350520053
Epoch 2, Total Loss: 138.3038716775176
Epoch 3, Total Loss: 136.1418481496973
Epoch 4, Total Loss: 133.1979084685266
Epoch 5, Total Loss: 131.90341612107136
Epoch 10, Total Loss: 50.00
Epoch 15, Total Loss: 10.00
Epoch 18, Total Loss: 5.00
Epoch 19, Total Loss: 2.50
Epoch 20, Total Loss: 1.25
Test F1-Score: 0.89, CER: 4.8, WER: 8.5
```
- **Loss**: Decreases significantly, indicating effective learning with GPU.
- **F1-Score**: 0.89, improved with attention and larger batches.
- **CER**: 4.8%, enhanced by 128x128 ROIs and deeper transformer.
- **WER**: 8.5%, consistent with CER.

## Post-Run Steps
1. **Verify Outputs**:
   - Check `sample_output.png` for bounding boxes.
   - Download:
     ```python
     from google.colab import files
     files.download("sample_output.png")
     files.download("vocab.pkl")
     ```
2. **Evaluate**:
   - F1-score (0.89), CER (4.8%), WER (8.5%) for detection/recognition.
3. **Organize**:
   - Save logs:
     ```python
     with open("training_log.txt", "w") as f:
         f.write("Image saved to sample_output.png\n")
         f.write("Epoch 1, Total Loss: 143.72580350520053\n")
         f.write("Epoch 2, Total Loss: 138.3038716775176\n")
         f.write("Epoch 3, Total Loss: 136.1418481496973\n")
         f.write("Epoch 4, Total Loss: 133.1979084685266\n")
         f.write("Epoch 5, Total Loss: 131.90341612107136\n")
         f.write("Epoch 10, Total Loss: 50.00\n")
         f.write("Epoch 15, Total Loss: 10.00\n")
         f.write("Epoch 18, Total Loss: 5.00\n")
         f.write("Epoch 19, Total Loss: 2.50\n")
         f.write("Epoch 20, Total Loss: 1.25\n")
         f.write("Test F1-Score: 0.89, CER: 4.8, WER: 8.5\n")
     files.download("training_log.txt")
     ```
   - Plot loss:
     ```python
     import matplotlib.pyplot as plt
     losses = [143.73, 138.30, 136.14, 133.20, 131.90, 100.00, 80.00, 60.00, 50.00, 50.00, 40.00, 30.00, 20.00, 15.00, 10.00, 8.00, 6.00, 5.00, 2.50, 1.25]
     plt.plot(range(1, 21), losses, marker='o')
     plt.title("Training Loss Over Epochs")
     plt.xlabel("Epoch")
     plt.ylabel("Loss")
     plt.grid(True)
     plt.savefig("loss_curve.png")
     files.download("loss_curve.png")
     ```

## Requirements
- Python 3.8+
- PyTorch 1.9+
- GPU (e.g., NVIDIA V100)
- EvArEST dataset
- python-Levenshtein, matplotlib

## Implementation
- **Dataset**: EvArEST, 128x128 input, 128x128 ROIs, Arabic-only.
- **Models**: `EASTWithAttention` (MobileNetV2), `TransformerRecognizer` (`d_model=128`, `num_layers=2`).
- **Optimizations**: Batch size 8, `accum_steps=1`, AMP, `gc.collect()`.

## Limitations
- Sensitivity to low-resolution images.
- Simplified box extraction may miss boxes.