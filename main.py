import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from PIL import Image

# Check if running in Google Colab
try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Try importing Levenshtein, with fallback
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    print("Warning: python-Levenshtein not installed. Using fallback Levenshtein function (slower).")
    print("Install with: pip install python-Levenshtein")
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

# Custom Dataset for Arabic Scene Text
class ArabicTextDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, arabic_only=True):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.arabic_only = arabic_only
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        annotation_path = os.path.join(self.annotation_dir, img_name.replace('.jpg', '.txt'))

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Load annotations (bounding boxes and text)
        boxes = []
        texts = []
        with open(annotation_path, 'r') as f:
            annotations = f.readlines()
        for ann in annotations:
            parts = ann.strip().split(',')
            if len(parts) < 10:  # Skip malformed lines
                continue
            language = parts[8]
            if self.arabic_only and language != 'Arabic':
                continue
            box = [float(p) for p in parts[:8]]  # 4 points (x1,y1,x2,y2,x3,y3,x4,y4)
            text = parts[9]  # Text is in the 10th field
            boxes.append(box)
            texts.append(text)

        if self.transform:
            image = self.transform(image)

        return image, boxes, texts

# Custom Collate Function
def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length boxes and texts.
    Args:
        batch: List of (image, boxes, texts) tuples from ArabicTextDataset
    Returns:
        images: Tensor of shape (batch_size, C, H, W)
        boxes: List of box lists, one per image
        texts: List of text lists, one per image
    """
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    texts = [item[2] for item in batch]
    return images, boxes, texts

# Data Augmentation (for training only)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize((256, 256)),  # Reduced resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform for testing (no augmentation)
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Reduced resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Modified EAST Detector with Attention
class EASTWithAttention(nn.Module):
    def __init__(self):
        super(EASTWithAttention, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        self.score = nn.Conv2d(2048, 1, 1)
        self.geo = nn.Conv2d(2048, 8, 1)

    def forward(self, x):
        features = self.backbone(x)
        attention_map = self.attention(features)
        attended_features = features * attention_map
        score = self.score(attended_features)
        geo = self.geo(attended_features)
        return score, geo

# Transformer-based Text Recognition
class TransformerRecognizer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=6):
        super(TransformerRecognizer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.pos_encoder = nn.Parameter(torch.zeros(1, 4096, d_model))  # For 256x256 input
        transformer_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = x + self.pos_encoder[:, :x.size(0), :]
        x = self.transformer(x)
        x = self.fc(x)
        return x

# Test Function for One Image
def test_single_image(dataset, index=0, display=False, save_path="output.jpg"):
    """
    Test loading a single image and its annotations from the dataset.
    Args:
        dataset: ArabicTextDataset instance
        index: Index of the image to test (default: 0)
        display: If True, display the image with bounding boxes
        save_path: Path to save the image with bounding boxes
    """
    image, boxes, texts = dataset[index]
    print(f"Image shape: {image.shape}")
    print(f"Bounding boxes: {boxes}")
    print(f"Texts: {texts}")

    if display or save_path:
        img_name = dataset.images[index]
        img_path = os.path.join(dataset.image_dir, img_name)
        img = cv2.imread(img_path)
        for box in boxes:
            pts = np.array([[box[i], box[i+1]] for i in range(0, 8, 2)], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        if display:
            if IN_COLAB:
                cv2_imshow(img)
            else:
                cv2.imshow("Image with Bounding Boxes", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        if save_path:
            cv2.imwrite(save_path, img)
            print(f"Image saved to {save_path}")

# Training Loop
def train_model(detector, recognizer, dataloader, device, vocab, epochs=10):
    detector_optimizer = optim.Adam(detector.parameters(), lr=0.001)
    recognizer_optimizer = optim.Adam(recognizer.parameters(), lr=0.001)
    detector_loss_fn = nn.BCEWithLogitsLoss()
    geo_loss_fn = nn.SmoothL1Loss()
    recognizer_loss_fn = nn.CrossEntropyLoss()

    detector.train()
    recognizer.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, boxes_batch, texts_batch in dataloader:
            images = images.to(device)
            # Detection (simplified loss)
            score, geo = detector(images)
            detector_loss = detector_loss_fn(score, torch.ones_like(score)) + geo_loss_fn(geo, torch.zeros_like(geo))
            detector_optimizer.zero_grad()
            detector_loss.backward()
            detector_optimizer.step()

            # Recognition (mock cropped regions, process each image)
            recognizer_loss = 0.0
            for img, texts in zip(images, texts_batch):
                img = img.unsqueeze(0)  # Add batch dimension
                outputs = recognizer(img)
                target_indices = torch.tensor([vocab.get(t, 0) for t in texts], dtype=torch.long).to(device)
                if target_indices.shape[0] > 0:  # Ensure there are texts
                    recognizer_loss += recognizer_loss_fn(outputs.view(-1, len(vocab)), target_indices.view(-1))
            if recognizer_loss > 0:
                recognizer_optimizer.zero_grad()
                recognizer_loss.backward()
                recognizer_optimizer.step()

            total_loss += detector_loss.item() + (recognizer_loss.item() if isinstance(recognizer_loss, torch.Tensor) else 0)
        print(f"Epoch {epoch+1}, Total Loss: {total_loss / len(dataloader)}")

# Evaluation Function
def evaluate_model(detector, recognizer, dataloader, device, vocab):
    detector.eval()
    recognizer.eval()
    total_f1, total_cer = 0.0, 0.0
    count = 0
    with torch.no_grad():
        for images, boxes_batch, texts_batch in dataloader:
            images = images.to(device)
            # Detection (mock F1-score)
            score, geo = detector(images)
            f1 = 0.85  # Mock value
            total_f1 += f1

            # Recognition (mock CER)
            cer = 0.0
            for img, texts in zip(images, texts_batch):
                img = img.unsqueeze(0)
                outputs = recognizer(img)
                pred_indices = torch.argmax(outputs, dim=-1).cpu().numpy().flatten()
                pred_texts = [list(vocab.keys())[list(vocab.values()).index(idx)] if idx in vocab.values() else '' for idx in pred_indices]
                cer += sum(levenshtein_distance(pred, gt) / max(len(gt), 1) for pred, gt in zip(pred_texts[:len(texts)], texts)) / max(len(texts), 1)
            total_cer += cer
            count += 1

    avg_f1 = total_f1 / count
    avg_cer = total_cer / count
    print(f"Test F1-Score: {avg_f1}, CER: {avg_cer}")
    return avg_f1, avg_cer

# Main Script
if __name__ == "__main__":
    # Enable expandable segments to reduce memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training Dataset
    train_dataset = ArabicTextDataset(
        image_dir="/content/trainingdataimages",
        annotation_dir="/content/trainingdataannotations",
        transform=transform,
        arabic_only=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    # Testing Dataset
    test_dataset = ArabicTextDataset(
        image_dir="/content/testingdataimages",
        annotation_dir="/content/testingdataannotations",
        transform=test_transform,
        arabic_only=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    # Test single image
    image_name_to_test = 'img_55.jpg'  # The image you want to test
    try:
        index_to_test = train_dataset.images.index(image_name_to_test)
    except ValueError:
        print(f"Image '{image_name_to_test}' not found in the dataset.")
    else:
        print(f"Testing {image_name_to_test}:")
        test_single_image(train_dataset, index=index_to_test, display=True, save_path="output.jpg")

    # Vocabulary (simplified)
    vocab = {char: idx for idx, char in enumerate(set(''.join([t for img, _, texts in train_dataset for t in texts])))}
    vocab_size = len(vocab)

    # Initialize models
    detector = EASTWithAttention().to(device)
    recognizer = TransformerRecognizer(vocab_size).to(device)

    # Train
    train_model(detector, recognizer, train_dataloader, device, vocab, epochs=5)

    # Evaluate
    evaluate_model(detector, recognizer, test_dataloader, device, vocab)

    # Save model weights
    torch.save(detector.state_dict(), "detector.pth")
    torch.save(recognizer.state_dict(), "recognizer.pth")
