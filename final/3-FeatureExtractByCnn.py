import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

weights_path = r'C:\Users\Hanieh\source\final\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(weights=weights_path, include_top=False)

model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)
image_folder = 'C:/Users/Hanieh/source/final/4-Fin-pics/' 

img_names = []
features_list = []

for img_name in os.listdir(image_folder):
    if img_name.endswith(".jpeg") or img_name.endswith(".png"):

        img_names.append(img_name)
        img_path = os.path.join(image_folder, img_name)
        
        img = load_img(img_path, target_size=(50, 50))
        img_array = img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0) 
        img_array = preprocess_input(img_array)
        
        features = model.predict(img_array)
        features_list.append(features.flatten())

features_array = np.array(features_list)
df = pd.DataFrame(features_list)

df['image_name'] = img_names
csv_file_path = '4-Fin-features.csv'

df.to_csv(csv_file_path, index=False)
print(f"Extracted features shape: {features_array.shape}")