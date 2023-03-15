import torch
import whisper

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "medium"
model = whisper.load_model(model_name).to(device)

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "whisper.onnx",
        opset_version=16,
        input_names=['input'],
        output_names=['output'])
