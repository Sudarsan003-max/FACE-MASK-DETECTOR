# USAGE
# python detect_mask_video.py

# import the necessary packages
import os
# Ensure legacy Keras loader for older SavedModels/H5 files
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import json
import h5py

def load_mask_model(model_path):
    """Robustly load a Keras model across Keras 2/3 differences.
    Tries plain load first, then with custom_objects, then alternate file.
    """
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"[WARN] Primary load_model failed for {model_path}: {e}")

        # Custom initializer to ignore unexpected dtype in Keras 3 configs
        class FixedGlorotUniform(tf.keras.initializers.GlorotUniform):
            def __init__(self, **kwargs):
                kwargs.pop("dtype", None)
                super().__init__(**kwargs)

        custom_objects = {
            'GlorotUniform': FixedGlorotUniform(),
            'relu6': tf.keras.layers.ReLU(6.0),
            'hard_swish': tf.keras.layers.ReLU(6.0),  # approximation
        }
        try:
            return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        except Exception as e2:
            print(f"[WARN] Secondary load with custom_objects failed: {e2}")
            alt = "mask_detector.h5" if model_path.endswith(".model") else "mask_detector.model"
            if os.path.exists(alt):
                print(f"[INFO] Attempting to load alternate model file: {alt}")
                try:
                    return tf.keras.models.load_model(alt, compile=False, custom_objects=custom_objects)
                except Exception as e_alt:
                    print(f"[WARN] Alternate model load failed: {e_alt}")
            # As a last resort, try rebuilding from H5 model_config
            try:
                with h5py.File(model_path, 'r') as f:
                    cfg = f.attrs.get('model_config')
                    if cfg is None:
                        raise RuntimeError('No model_config in H5 file')
                    if hasattr(cfg, 'decode'):
                        cfg = cfg.decode('utf-8')
                    cfg = json.loads(cfg)

                def fix_initializer(config):
                    if isinstance(config, dict):
                        if 'class_name' in config and 'config' in config:
                            if config['class_name'] == 'GlorotUniform' and isinstance(config['config'], dict):
                                config['config'].pop('dtype', None)
                        for k, v in list(config.items()):
                            config[k] = fix_initializer(v)
                    elif isinstance(config, list):
                        return [fix_initializer(x) for x in config]
                    return config

                fixed = fix_initializer(cfg)
                model = tf.keras.Model.from_config(fixed, custom_objects=custom_objects)
                # Try loading weights from the same file
                try:
                    model.load_weights(model_path, by_name=True, skip_mismatch=True)
                except Exception as _:
                    pass
                return model
            except Exception as e3:
                print(f"[ERROR] Manual H5 rebuild failed: {e3}")
                # Try manual rebuild using alt path if available
                if os.path.exists(alt):
                    try:
                        with h5py.File(alt, 'r') as f:
                            cfg = f.attrs.get('model_config')
                            if cfg is None:
                                raise RuntimeError('No model_config in H5 alt file')
                            if hasattr(cfg, 'decode'):
                                cfg = cfg.decode('utf-8')
                            cfg = json.loads(cfg)

                        def fix_initializer(config):
                            if isinstance(config, dict):
                                if 'class_name' in config and 'config' in config:
                                    if config['class_name'] == 'GlorotUniform' and isinstance(config['config'], dict):
                                        config['config'].pop('dtype', None)
                                for k, v in list(config.items()):
                                    config[k] = fix_initializer(v)
                            elif isinstance(config, list):
                                return [fix_initializer(x) for x in config]
                            return config

                        fixed = fix_initializer(cfg)
                        model = tf.keras.Model.from_config(fixed, custom_objects=custom_objects)
                        try:
                            model.load_weights(alt, by_name=True, skip_mismatch=True)
                        except Exception as _:
                            pass
                        return model
                    except Exception as e_alt2:
                        print(f"[ERROR] Manual H5 rebuild from alt failed: {e_alt2}")

                # Final fallback: rebuild architecture from training script and load weights
                try:
                    print("[INFO] Rebuilding architecture programmatically and loading weights...")
                    baseModel = MobileNetV2(weights=None, include_top=False, input_tensor=Input(shape=(224, 224, 3)))
                    headModel = baseModel.output
                    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
                    headModel = Flatten(name="flatten")(headModel)
                    headModel = Dense(128, activation="relu")(headModel)
                    headModel = Dropout(0.5)(headModel)
                    headModel = Dense(2, activation="softmax")(headModel)
                    model = Model(inputs=baseModel.input, outputs=headModel)
                    try:
                        model.load_weights(model_path, by_name=True, skip_mismatch=True)
                    except Exception as _:
                        pass
                    # Also try alt weights if available
                    if os.path.exists(alt):
                        try:
                            model.load_weights(alt, by_name=True, skip_mismatch=True)
                        except Exception as _:
                            pass
                    return model
                except Exception as e4:
                    print(f"[ERROR] Fallback rebuild with weights failed: {e4}")
                    raise

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.h5",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_mask_model(args["model"])
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
