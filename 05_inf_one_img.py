import cv2
import torch


def main():
    img = cv2.imread("b.bmp")
    model_path = "scripted_pidnet_model_2022-10-14.pt"
    input_size = 384

    # build model
    model = torch.jit.load(model_path)
    model.eval()

    resized_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    main()