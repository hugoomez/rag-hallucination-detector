import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
