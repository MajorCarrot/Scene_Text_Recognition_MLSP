# Copyright (C) 2021 Adithya Venkateswaran
#
# MLSP_project is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MLSP_project is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MLSP_project. If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from CRAFT import craft, craft_utils, file_utils, imgproc
from LangaugeModel import model
from LangaugeModel.dataset import AlignCollate, RawDataset
from LangaugeModel.utils import AttnLabelConverter, CTCLabelConverter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALLOWED_IMAGE_FORMATS = [".jpg", ".png", ".jpeg"]
files = []


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(
    net, image, text_threshold, link_threshold, low_text, cuda, poly, args
):
    t0 = time.time()

    # resize
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image,
        args.canvas_size,
        interpolation=cv2.INTER_LINEAR,
        mag_ratio=args.mag_ratio,
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = torch.autograd.Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys, det_scores = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text, det_scores


def drawBoundingBoxes(imageData, imageOutputPath, rectangles, color):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    rectangles: list of rectangles to be plotted, each rectangle is a tuple
        (coords, label)
    color: Bounding box color candidates, list of RGB tuples.
    """
    for res in rectangles:
        coords = np.array(res[0], np.int32)
        label = res[1]
        left, top = res[0][0]

        left = int(left)
        top = int(top)

        coords.reshape((-1, 1, 2))

        imgHeight, imgWidth, _ = imageData.shape

        thick = int((imgHeight + imgWidth) // 900)
        # print(left, top, right, bottom)

        cv2.polylines(imageData, [coords], True, color, thick)
        cv2.putText(
            imageData,
            label,
            (left, top - 12),
            0,
            1e-3 * imgHeight,
            color,
            thick // 3,
        )
    cv2.imwrite(imageOutputPath, imageData)


def crop(pts, image):
    """
    Takes inputs as 8 points
    and Returns cropped, masked image with a white background
    """
    pts = pts.astype(int)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = image[y : y + h, x : x + w].copy()
    pts = pts - pts.min(axis=0)
    # print(pts.shape)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst2


# CRAFT
parser = argparse.ArgumentParser(description="CRAFT Text Detection")
parser.add_argument(
    "--trained_model",
    default="./weights/craft_mlt_25k.pth",
    type=str,
    help="pretrained model",
)
parser.add_argument(
    "--text_threshold",
    default=0.7,
    type=float,
    help="text confidence threshold",
)
parser.add_argument(
    "--low_text", default=0.4, type=float, help="text low-bound score"
)
parser.add_argument(
    "--link_threshold",
    default=0.4,
    type=float,
    help="link confidence threshold",
)
parser.add_argument(
    "--cuda", default=True, type=str2bool, help="Use cuda for inference"
)
parser.add_argument(
    "--canvas_size", default=1280, type=int, help="image size for inference"
)
parser.add_argument(
    "--mag_ratio", default=1.5, type=float, help="image magnification ratio"
)
parser.add_argument(
    "--poly", default=False, action="store_true", help="enable polygon type"
)
parser.add_argument(
    "--show_time",
    default=False,
    action="store_true",
    help="show processing time",
)
parser.add_argument(
    "--test_folder",
    default="./input/",
    type=str,
    help="folder path to input images",
)
parser.add_argument(
    "--output_folder",
    default="./output/",
    type=str,
    help="folder path to output images",
)
parser.add_argument(
    "--Transformation",
    type=str,
    required=True,
    help="Transformation stage. None|TPS",
)
parser.add_argument(
    "--FeatureExtraction",
    type=str,
    required=True,
    help="FeatureExtraction stage. VGG|RCNN|ResNet",
)
parser.add_argument(
    "--SequenceModeling",
    type=str,
    required=True,
    help="SequenceModeling stage. None|BiLSTM",
)
parser.add_argument(
    "--Prediction",
    type=str,
    required=True,
    help="Prediction stage. CTC|Attn",
)
parser.add_argument(
    "--nlp_model",
    required=True,
    help="path to saved_model to evaluation",
)
parser.add_argument(
    "--workers", type=int, help="number of data loading workers", default=4
)
parser.add_argument(
    "--batch_size", type=int, default=192, help="input batch size"
)
parser.add_argument(
    "--batch_max_length", type=int, default=25, help="maximum-label-length"
)
parser.add_argument(
    "--imgH", type=int, default=32, help="the height of the input image"
)
parser.add_argument(
    "--imgW", type=int, default=100, help="the width of the input image"
)
parser.add_argument("--rgb", action="store_true", help="use rgb input")
parser.add_argument(
    "--character",
    type=str,
    default="0123456789abcdefghijklmnopqrstuvwxyz",
    help="character label",
)
parser.add_argument(
    "--sensitive", action="store_true", help="for sensitive character mode"
)
parser.add_argument(
    "--PAD",
    action="store_true",
    help="whether to keep ratio then pad for image resize",
)
parser.add_argument(
    "--num_fiducial",
    type=int,
    default=20,
    help="number of fiducial points of TPS-STN",
)
parser.add_argument(
    "--input_channel",
    type=int,
    default=1,
    help="the number of input channel of Feature extractor",
)
parser.add_argument(
    "--output_channel",
    type=int,
    default=512,
    help="the number of output channel of Feature extractor",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    help="the size of the LSTM hidden state",
)
args = parser.parse_args()

if __name__ == "__main__":
    data = pd.DataFrame(columns=["image_name", "word_bboxes", "pred_words"])

    print("Fetching file names")
    for file in sorted(os.listdir(args.test_folder)):
        if os.path.splitext(file)[1] in ALLOWED_IMAGE_FORMATS:
            files.append(os.path.join(args.test_folder, file))

    files = files

    if not os.path.isdir(args.output_folder):
        try:
            os.makedirs(args.output_folder)
        except OSError:
            print(f"Couldn't create the output folder at {args.output_folder}")
            raise

    net = craft.CRAFT()

    print(f"Loading CRAFT weights from {args.trained_model}")
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(
            copyStateDict(torch.load(args.trained_model, map_location="cpu"))
        )

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
        args.num_gpu = torch.cuda.device_count()

    net.to(device)
    aligncollate = AlignCollate(args.imgH, args.imgW, args.PAD)

    if "CTC" in args.Prediction:
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)

    args.num_class = len(converter.character)

    nlp_net = model.Model(args)
    print(f"Loading NLP weights from {args.nlp_model}")
    state_dict = torch.load(args.nlp_model, map_location=device)

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    nlp_net.load_state_dict(new_state_dict)

    nlp_net.to(device)

    t = time.time()
    for k, image_path in enumerate(files):
        print(f"Processing image: {k+1}/{len(files)} at {image_path}")
        image = imgproc.loadImage(image_path)
        bboxes, polys, score_text, det_scores = test_net(
            net,
            image,
            args.text_threshold,
            args.link_threshold,
            args.low_text,
            args.cuda,
            args.poly,
            args,
        )

        bbox_score = {}

        for box_num in range(len(bboxes)):
            key = str(det_scores[box_num])
            item = bboxes[box_num]
            bbox_score[key] = item

        # data["word_bboxes"][k] = bbox_score

        # save score text
        print("Saving the image and mask")
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(args.output_folder, filename + "_mask.jpg")
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(
            image_path, image[:, :, ::-1], polys, dirname=args.output_folder
        )

        rectangles = []
        crop_folder = os.path.join(args.output_folder, filename)

        saved_coords = {}
        if not os.path.isdir(crop_folder):
            os.makedirs(crop_folder)
        for crop_no, (confidence, coords) in enumerate(bbox_score.items()):
            coords[coords < 0] = 0
            print(
                f"Found text, confidence: {confidence}, coords: {list(coords)}"
            )
            cropped_image = crop(coords, image)
            crop_path = os.path.join(
                crop_folder, filename + f"_crop_{crop_no}.jpg"
            )
            cv2.imwrite(crop_path, cropped_image)
            saved_coords[crop_path] = coords
            print(f"Saved to {crop_path}")

        demo_data = RawDataset(root=crop_folder, opt=args)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=aligncollate,
            pin_memory=True,
        )
        nlp_net.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                print("Here!!")
                batch_size = image_tensors.size(0)
                cropped_image = image_tensors.to(device)

                length_for_pred = torch.IntTensor(
                    [args.batch_max_length] * batch_size
                ).to(device)
                text_for_pred = (
                    torch.zeros((batch_size, args.batch_max_length + 1))
                    .long()
                    .to(device)
                )

                if "CTC" in args.Prediction:
                    preds = nlp_net(cropped_image, text_for_pred)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, preds_size)
                else:
                    preds = nlp_net(
                        cropped_image, text_for_pred, is_train=False
                    )
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, 2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(
                    image_path_list, preds_str, preds_max_prob
                ):
                    if "Attn" in args.Prediction:
                        pred_EOS = pred.find("[s]")
                        pred = pred[:pred_EOS]
                        pred_max_prob = preds_max_prob[:pred_EOS]

                    confidence_score = pred_max_prob.cumprod(0)[-1]
                    coords = saved_coords[img_name]

                    print(f"pred: {pred}, confidence: {confidence_score:0.4f}")
                    rectangles.append(
                        (coords, f"{pred} {confidence_score:0.4f}")
                    )
                    data = data.append(
                        {
                            "image_name": filename,
                            "word_bboxes": coords,
                            "pred_words": pred,
                        },
                        ignore_index=True,
                    )

        image_output_path = os.path.join(
            args.output_folder,
            os.path.splitext(os.path.split(image_path)[1])[0]
            + "_labelled.jpg",
        )
        drawBoundingBoxes(
            image, image_output_path, rectangles, color=(255, 255, 255)
        )
        print("")

    data.to_csv("./data.csv", sep=",", na_rep="Unknown")
    print("Average time : {}s".format((time.time() - t) / len(files)))
