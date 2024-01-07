import gradio as gr
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision



# 加载与训练中使用的相同结构的模型
def load_model(checkpoint_path, num_classes):
    # 加载预训练的ResNet50模型
    try:
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False

    if torch.cuda.is_available():
        device = "cuda"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"

    model = torchvision.models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model


# 加载图像并执行必要的转换的函数
def process_image(image, image_size):
    # Define the same transforms as used during training
    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocessing(image).unsqueeze(0)
    return image


# 预测图像类别并返回概率的函数
def predict(image):
    classes = {'0': 'cat', '1': 'dog'}  # Update or extend this dictionary based on your actual classes
    image = process_image(image, 256)  # Using the image size from training
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1).squeeze()  # Apply softmax to get probabilities
    # Mapping class labels to probabilities
    class_probabilities = {classes[str(i)]: float(prob) for i, prob in enumerate(probabilities)}
    return class_probabilities


# 定义到您的模型权重的路径
checkpoint_path = 'checkpoint/latest_checkpoint.pth'
num_classes = 2
model = load_model(checkpoint_path, num_classes)

# 定义Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
    title="Cat vs Dog Classifier",
    examples=["test_images/test_cat.jpg", "test_images/test_dog.jpg"]
)

if __name__ == "__main__":
    iface.launch()