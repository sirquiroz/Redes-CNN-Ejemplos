import argparse, keras, os, cv2, glob, numpy as np
from keras.models import model_from_json
parser = argparse.ArgumentParser()
parser.add_argument("image", help="Image or folder of images to predict",
                    type=str)
args = parser.parse_args()

# Cargamos el modelo
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

# Cargamos la imagen/imagenes
if os.path.isdir(args.image):
	path_images = []
	for ext in ['jpg', 'jpeg', 'tiff', 'png', 'gif']:
		path_images += glob.glob(os.path.join(args.image, "*." + ext))
else:
	path_images = [args.image]

images = np.empty(shape=(len(path_images), 32, 32, 3))
real_images = []
for i, path_im in enumerate(path_images):
	im = cv2.imread(path_im)
	real_images.append(im)
	images[i] = np.float32(cv2.resize(im, (32, 32))[..., ::-1]/255.0)

raw_results = model.predict(images)
results = {}
labels = {0: 'cat', 1: 'dog'}
text_to_show = {'cat': 'meowww', 'dog': 'bupp'}

def drawText(im, text, color):
	text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
	text_pos_x = (im.shape[1] - text_size[0]) // 2
	text_pos_y = (im.shape[0] + text_size[1]) // 2

	cv2.putText(im, text, (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

for i in range(raw_results.shape[0]):
	if(raw_results[i] > 0.5): # Revisar en el Labels de training
		drawText(real_images[i], text_to_show[labels[1]], (0, 255, 0))
		results[path_images[i]] = (labels[1], raw_results[i])
	else:
		drawText(real_images[i], text_to_show[labels[0]], (0, 0, 255))
		results[path_images[i]] = (labels[0], 1 - raw_results[i])

	cv2.imshow('Image', real_images[i])
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
for path_im, (name, confidence) in results.items():
	print('{0} = {1} ({2})'.format(path_im, name, confidence))