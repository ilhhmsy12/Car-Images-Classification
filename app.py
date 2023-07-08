from flask import Flask, render_template, request
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
import keras.utils as image
#from PIL import image

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
  imagefile = request.files['flowerimage']
  image_path = "static/" + imagefile.filename
  imagefile.save(image_path)

  IMAGE_SIZE = (128,128)
  BATCH_SIZE = 32

  model = keras.models.load_model('model_cars.h5')
  img = image.load_img(image_path, target_size=IMAGE_SIZE)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
 
  images = np.vstack([x])
  classes = model.predict(images, batch_size=BATCH_SIZE)
  classes = np.argmax(classes)
  if classes==0:
    result = "Audi"
    print('Audi')
  elif classes==1:
    result = "Hyundai Creta"
    print('Hyundai Creta')
  elif classes==2:
    result = "Mahindra Scorpio"
    print('Mahindra Scorpio')
  elif classes==3:
    result = "Rolls Royce"
    print('Rolls Royce')
  elif classes==4:
    result = "Swift"
    print('Swift')
  elif classes==5:
    result = "Tata Safari"
    print('Tata Safari')
  else:
    result = "Toyota Innova"
    print('Toyota Innova')

  return render_template('index.html', prediction=result, img_file=image_path)

if __name__ == "__main__":
  app.run(debug=True)
