# Team: Nienasycone Gradienty

This repo contains (partial) solutions to the #1 SKYHACKS Hackhaton.


## Installation

1. Create virtual environment:

	`conda create -n skyhacks python=3.6`

2. Install requirements:

      `pip install -r requrirements.txt`

## Data set creation

There are two scripts under `hakaton/sandbox/resize_*.py` to resize the images to 640x512 and 320x256 pixels and change them to grayscale.

### Attribution

For attribution we've used https://github.com/tzutalin/labelImg.

This tool is able to return data in two formats: yolo and pascal voc. We've used and parsed the second one. The script to parse these files is located under `hakaton/sandbox/parse_pascal_voc_files.py`

## Approach

We wanted to create three different solutions of three challenges:

1. For wagon detection, we use a rather shallow DNN classifier which detects:
a) empty image
b) gap between subsequent cars
c) beginning / end of the train

Proposed Model for detection gaps between wagons is located inside package: "hakaton/model/wagon_detector_model.py".

2. For UIC code detection, we've ended up with a similar approach, although the plan was to use the YOLO network architecture and detect the boundaries between the UIC code is located. Such boundaries could be used to feed data into the third model.

We also try more simple solution. We trained classificator which try recognise pictures with uic code.
Model located inside package "hakaton/model/uic_detector_model.py".

3. For UIC code attribution, we wanted to perform a OCR based on the window chosen by the 2nd model. Unfortunately, the 2nd approach has failed ;).
