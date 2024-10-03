import torch
from torch import nn

class FashionMNISTModelV0(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.convo_block_1 = nn.Sequential(
        nn.Conv2d(in_channels = input_shape,
                  out_channels = hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
         nn.ReLU(),
         nn.Conv2d(in_channels = hidden_units,
                   out_channels = hidden_units,
                   kernel_size=3,
                   stride=1,
                   padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.convo_block_2 = nn.Sequential(
        nn.Conv2d(in_channels = hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7,
                  out_features=output_shape)
    )

  def forward(self , X:torch.tensor):
    X = self.convo_block_1(X)
    X = self.convo_block_2(X)
    X = self.classifier(X)
    return X
  

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_prediction(model: torch.nn.Module,
                           img: torch.tensor
                           ):
  """
  Makes a single prediction on a single image
  returns value from 0 to 9 which represents a label class 
  of the data. Refer to data for more clarity
  """
  pred_logits = model(img.to(device).unsqueeze(dim = 0)).to(device)
  pred_probs = pred_logits.argmax(dim = 1)
  return pred_probs