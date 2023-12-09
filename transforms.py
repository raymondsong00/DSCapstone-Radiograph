from torchvision import transforms

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=[-15,15], translate=(0.05,0.1), scale=(0.9, 1.1)),
        transforms.Resize(256,antialias=False), #224
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([
        transforms.Resize(256,antialias=False),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return train_transforms, test_transforms
