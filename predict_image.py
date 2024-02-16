# make predictions on images using ensemble of model_1 and model_2

from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf

model_1 = load_model('nsfw_model_1.h5')
model_2 = load_model('nsfw_model_2.h5')

def predict_image(img_src):
    with tf.device('/gpu:0'):
        img1 = load_img(img_src, target_size=(500, 500))
        img2 = load_img(img_src, target_size=(400, 400))

        x1 = img_to_array(img1)
        x1 = np.expand_dims(x1, axis=0)
        x1 = x1/255

        x2 = img_to_array(img2)
        x2 = np.expand_dims(x2, axis=0)
        x2 = x2/255

        pred1 = model_1.predict(x1, verbose=0)
        pred2 = model_2.predict(x2, verbose=0)

        pred = (pred1 + pred2)/2

        return pred

# for testing
if __name__ == '__main__':
    labels = ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy']
    img_src = f'test_images/img3.jpg'

    pred = predict_image(img_src)
    print(f"img is ", labels[np.argmax(pred)])
      