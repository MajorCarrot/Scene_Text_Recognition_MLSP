<!--
 Copyright (C) 2021 Adithya Venkateswaran
 
 final_project is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 final_project is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 You should have received a copy of the GNU Lesser General Public License
 along with final_project. If not, see <http://www.gnu.org/licenses/>.
-->

# Scene Text Recognition

## Usage Instructions
Open MLSP_Project_Test.ipynb in a colab environment and run all the cells.

If you want to manually run the scene text recognition model, follow the following steps:

1. Clone this repo `git clone https://github.com/MajorCarrot/Scene_Text_Recognition_MLSP.git ./STR/`

1. Change your working directory to the root directory of the cloned repo `cd ./STR`

1. Create a folder called input `mkdir input` and add your test images to the input folder

1. Download the weights for the CRAFT and nlp models from [this drive folder](https://drive.google.com/drive/folders/1rQgeLmhN_88Ut_9q0V7DAOAgYcxyVe3P?usp=sharing). Make a new folder under STR called weights and copy the weights there

1. Run `pip install -r requirements.txt` to ensure that all the required files are loaded into your system (It is highly recommended that this be done in a virtualenvironment. Refer to [conda](https://docs.conda.io/en/latest/) or [python venv](https://docs.python.org/3/library/venv.html) for more info)

1. If you have a system with GPU, run `python detect_text.py --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC --nlp_model ./weights/None-VGG-BiLSTM-CTC.pth`, else run `python detect_text.py --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC --nlp_model ./weights/None-VGG-BiLSTM-CTC.pth --cuda False`

1. Voila, the transformed outputs are there in the outputs folder. Images with `_labelled` at the end of their names have the predicted words with confidences overlayed on the image. The masks are also available in the same folder.

## More info

To learn more about how this works, refer to our [presentation](https://docs.google.com/presentation/d/1g2gOzDQ1wt8Tsz3_JSdpkUgWkaNGSuBCuFWwqPiPSZk/edit?usp=sharing)

## Pre-run Colab Notebooks
1. [Initial test of CRAFT](https://drive.google.com/file/d/1c5t3JszLOCWxNlOltDwy_jSkxZnsRCcn/view?usp=sharing)

1. [Tests with YOLOv2](https://drive.google.com/file/d/1i7ShPH6EnGO9bKC3lTG-Aj1_mWx_O20t/view?usp=sharing)

1. [Results analysis code for CRAFT (initial version)](https://colab.research.google.com/drive/1IIDlCR3cUD-0cbdDaYzhbyF4uE6Pj78S?usp=sharing)

1. [Final test of CRAFT with this repo](https://colab.research.google.com/drive/1_6i4BlcuDOxbFoRU6nk-5OxAteLrBa-y?usp=sharing)

## Check out the Results

You can view and download the results from [this drive folder](https://drive.google.com/drive/folders/1IlpWWv5qd3J4-qH-7yc64kZW48zqQCWB?usp=sharing)

## Runtime statistics
On GPU, it takes an average of 0.5 seconds for the image to be processed, text extracted and output predicted
