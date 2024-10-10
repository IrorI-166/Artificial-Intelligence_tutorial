#このへんはPyTorchドキュメントのNNチュートリアルを参考にします
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    #この構文少し分からない
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #最初の次元以外の次元を平坦化（一次元化） #NNにおける次元→入力データの次元のこと
        self.linear_relu_stack = nn.Sequential( #モデルのコンテナを作成、コンテナに渡されたモジュールの順にモデルが作成される？
            nn.Linear(28 * 28, 512), #線形変換モジュールを使ってinput size（第1引数）とoutput size（第2引数）の定義
            nn.ReLU(), #非線形の活性化関数を用いて非線形化
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

        def foward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

model = NeuralNetwork().to(device)
print(model)