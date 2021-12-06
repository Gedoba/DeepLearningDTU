from flask import Flask, request
import torch
from data_generator import make_dataloaders
from model import MainModel
from train import  load_model, build_backbone_unet
from visualize import resize_np_array_image, to_rgb
from flask import Flask, make_response, request, render_template
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    net_G = build_backbone_unet(n_input=1, n_output=2, size=256, backbone_name="resnet34")
    net_G.load_state_dict(torch.load("resnet34-unet_noBG_YCbCr_FULL.pt", map_location=device))
    loaded_model = MainModel(net_G=net_G, L1LossType="L1Loss", ganloss="lsgan")
    load_model("./final_models/model_pretrained_noBG_part1_resnet34_YCbCr_L1Loss_lsgan_FULL.pt", loaded_model) 
    return loaded_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)
model = get_model()
model.eval()

def get_prediction(image_bytes):
    test_dl = make_dataloaders(batch_size=1, bytes=image_bytes, split='val', color_space="YCbCr")
    img = next(iter(test_dl))
    width, height = test_dl.dataset.sizes[0]
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(img)
        model.forward()
    fake_color = model.fake_color.detach()
    known_channel = model.known_channel
    fake_imgs = to_rgb(known_channel, fake_color, "YCbCr")
    fake_img = fake_imgs[0]
    fake_img = resize_np_array_image(fake_img, height, width)
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', fake_img * 255)
    response = make_response(buffer.tobytes())
    response.headers.set('Content-Type', 'image/png')
    response.headers.set(
        'Content-Disposition', 'attachment', filename='prediction.png')
    return response

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files["file"]
        img_bytes = file.read()
        return get_prediction(image_bytes=img_bytes)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')