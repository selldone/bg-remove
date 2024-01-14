import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms.functional import normalize

from service.isnet.data_loader_cache import im_preprocess
from service.isnet import model_isnet


class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image


transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])


def load_image(in_memory_file, hypar):
    input_image = imageio.imread(in_memory_file)

    # Assuming im_preprocess is a custom function that returns the preprocessed image and its shape
    processed_image, image_shape = im_preprocess(input_image, hypar["cache_size"])

    # Convert the processed image to a tensor and divide by 255.0 if needed
    image_tensor = torch.divide(torch.tensor(processed_image), 255.0)

    # Convert the shape to a torch tensor
    shape_tensor = torch.from_numpy(np.array(image_shape))

    # Apply the transform and make a batch of the image
    transformed_image = transform(image_tensor).unsqueeze(0)

    return transformed_image, shape_tensor.unsqueeze(0)


def build_model(hypar, device):
    net = hypar["model"]  # GOSNETINC(3,1)

    # convert to half precision
    if (hypar["model_digit"] == "half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if (hypar["restore_model"] != ""):
        net.load_state_dict(torch.load(hypar["model_path"] + "/" + hypar["restore_model"], map_location=device))
        net.to(device)
    net.eval()
    return net


def predict(net, inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    net.eval()

    if (hypar["model_digit"] == "full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device)  # wrap inputs in Variable

    ds_val = net(inputs_val_v)[0]  # list of 6 results

    pred_val = ds_val[0][0, :, :, :]  # B x 1 x H x W    # we want the first one which is the most accurate prediction

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(
        F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)  # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)  # it is the mask we need


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

hypar = {}  # paramters for inferencing

hypar["model_path"] = "./service/isnet/model"  ## load trained weights from this path
hypar["restore_model"] = "isnet-general-use.pth"  ## name of the to-be-loaded weights
hypar["interm_sup"] = True  ## indicate if activate intermediate feature supervision

##  choose floating point accuracy --
hypar["model_digit"] = "full"  ## indicates "half" or "full" accuracy of float number
hypar["seed"] = 0

hypar["cache_size"] = [1024, 1024]  ## cached input spatial resolution, can be configured into different size

## data augmentation parameters ---
hypar["input_size"] = [1024,
                       1024]  ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
hypar["crop_size"] = [1024,
                      1024]  ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation

hypar["model"] = model_isnet.ISNetDIS()

net = build_model(hypar, device)


def remove_bg_mult(in_memory_file):

    image_tensor, orig_size = load_image(in_memory_file, hypar)

    mask = predict(net, image_tensor, orig_size, hypar, device)

    # Convert the mask to a PIL Image
    mask_image = convert_to_image(mask)

    # Open the original image
    original_image = Image.open(in_memory_file)

    # Resize mask to match original image size
    mask_image = mask_image.resize(original_image.size, Image.BILINEAR)

    # Convert mask image to 'L' mode (grayscale)
    mask_image = mask_image.convert("L")

    # Apply the mask to the original image
    result_image = Image.composite(original_image, Image.new("RGBA", original_image.size), mask_image)

    return result_image


def convert_to_image(output_array):
    # Assuming output_array is a 2D NumPy array with values in the range [0, 255]
    # Convert the NumPy array to an image
    image = Image.fromarray(output_array.astype(np.uint8))

    return image
