# PyTorch counterpart of ../a3c_visualize.py. Inspection-only: loads a
# checkpoint and plots the first conv layer's filters so you can eyeball
# what the network learned to detect.
import matplotlib.pyplot as plt
import torch

from model import GameACFFNetwork, GameACLSTMNetwork

from constants import ACTION_SIZE, USE_LSTM
from train import _find_latest_checkpoint


def main():
  if USE_LSTM:
    model = GameACLSTMNetwork(ACTION_SIZE)
  else:
    model = GameACFFNetwork(ACTION_SIZE)

  checkpoint_path = _find_latest_checkpoint()
  if checkpoint_path:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print("checkpoint loaded:", checkpoint_path)
  else:
    print("Could not find old checkpoint")
  model.eval()

  # conv1.weight shape is (out_channels=16, in_channels=4, 8, 8) -- note
  # this is PyTorch's (out, in, h, w) layout, the transpose of the TF
  # version's (h, w, in, out) W_conv1
  W_conv1 = model.trunk.conv1.weight.detach().numpy()

  # show graph of W_conv1 -- one small grayscale image per (input channel,
  # output filter) pair, so a 4x16 grid of 8x8 patches in total
  fig, axes = plt.subplots(4, 16, figsize=(12, 6),
               subplot_kw={'xticks': [], 'yticks': []})
  fig.subplots_adjust(hspace=0.1, wspace=0.1)

  for ax, i in zip(axes.flat, range(4 * 16)):
    inch = i // 16
    outch = i % 16
    img = W_conv1[outch, inch, :, :]
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title(str(inch) + "," + str(outch))

  plt.show()


if __name__ == "__main__":
  main()
