
import torch

def one_hot_encoding(semantic, num_classes=20):
    one_hot = torch.zeros(num_classes, semantic.size(1), semantic.size(2))
    for class_id in range(num_classes):
        one_hot[class_id, :, :] = (semantic.squeeze(0) == class_id)
    one_hot = one_hot[:num_classes-1, :, :]
    return one_hot
