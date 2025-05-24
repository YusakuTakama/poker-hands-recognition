import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from app.card import Card
from .card_mapper import create_card_from_string
from ultralytics import YOLO
from PIL import Image
from transformers import ViTForImageClassification, ViTConfig
# pil to tensor
from torchvision.transforms import ToTensor


index_to_label = {
    0: '1c',
    1: '1d',
    2: '1h',
    3: '1s',
    4: '2c',
    5: '2d',
    6: '2h',
    7: '2s',
    8: '3c',
    9: '3d',
    10: '3h',
    11: '3s',
    12: '4c',
    13: '4d',
    14: '4h',
    15: '4s',
    16: '5c',
    17: '5d',
    18: '5h',
    19: '5s',
    20: '6c',
    21: '6d',
    22: '6h',
    23: '6s',
    24: '7c',
    25: '7d',
    26: '7h',
    27: '7s',
    28: '8c',
    29: '8d',
    30: '8h',
    31: '8s',
    32: '9c',
    33: '9d',
    34: '9h',
    35: '9s',
    36: 'jc',
    37: 'jd',
    38: 'jh',
    39: 'js',
    40: 'kc',
    41: 'kd',
    42: 'kh',
    43: 'ks',
    44: 'qc',
    45: 'qd',
    46: 'qh',
    47: 'qs',
    48: 'tc',
    49: 'td',
    50: 'th',
    51: 'ts'}


class CardDetectorInterface:
    def __init__(self):
        raise NotImplementedError("This class should not be instantiated directly. Use a subclass instead.")

    def detect(self, image) -> List[Card]:
        """Detect cards in the given image and return a list of Card objects."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class MyViTDetector(nn.Module):
    def __init__(self, num_classes=52):
        super(MyViTDetector, self).__init__()
        config = ViTConfig.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=num_classes)
        self.model = ViTForImageClassification(config)

    def forward(self, x):
        return self.model(x)


class YoLoViTDetector(CardDetectorInterface):
    def __init__(self):
        self.card_ditector = YOLO('./pretrain_weight/best.pt')
        self.classifier = MyViTDetector(num_classes=52)
        self.classifier.load_state_dict(torch.load("pretrain_weight/epoch170_step511_acc=92.31.pt")['model_state_dict'])

    def detect(self, image: Image) -> List[Card]:
        card_bboxes = self.card_ditector(image)
        output_data = card_bboxes[0].boxes.data
        num_bboxes = output_data.shape[0]
        if num_bboxes < 7:
            return None
        hands_bboxes = (card_bboxes[0].boxes.xyxyn[:7])  # shape: (7, 4(xyxy normalized))
        tensor_img = ToTensor()(image)
        cropped_imgs = []

        for i in range(7):
            hand_bbox = hands_bboxes[i]
            cropped_img = tensor_img[:, int(hand_bbox[1] * tensor_img.shape[1]):int(hand_bbox[3] * tensor_img.shape[1]),
                                     int(hand_bbox[0] * tensor_img.shape[2]):int(hand_bbox[2] * tensor_img.shape[2])]
            resized = F.interpolate(cropped_img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            cropped_imgs.append(resized)
        cropped_imgs_tensor = torch.cat(cropped_imgs, dim=0)

        with torch.no_grad():
            logits = self.classifier(cropped_imgs_tensor).logits
            probabilities = F.softmax(logits, dim=1)
            _, top_indices = torch.topk(probabilities, k=1, dim=1)

        cards = []
        for index in top_indices.squeeze().tolist():
            card_str = index_to_label[index]
            card = create_card_from_string(card_str)
            cards.append(card)
        return cards
