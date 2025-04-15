# An MRI based Early Detection of Alzheimer's Disease Using VGG16 as Feature Extractor

This project involves using Convolutional Neural Networks (CNN) for detecting Alzheimer's disease from medical images. We use the VGG16 model pre-trained on ImageNet, fine-tuned for this specific task of Alzheimer's diagnosis. The model classifies images into four categories:

- Mild Dementia
- Moderate Dementia
- Non Dementia
- Very Mild Dementia

The project also includes a Streamlit app that allows users to upload images and get predictions on the Alzheimer's disease stage.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- PIL (Pillow)
- Streamlit

You can install the required libraries using `pip`:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn pillow streamlit
```
## Model File  
You can download the trained model file from the link below:  
https://drive.google.com/drive/folders/1HRa__luLXWSPivx9g60vKjkJJgRsln_p?usp=drive_link
## DataSet
The dataset for the project should be organized into directories as follows:
```
project/
│
├── train/
│   ├── MildDemented/
│   ├── ModerateDemented/
│   ├── NonDemented/
│   └── VeryMildDemented/
│
├── test/
│   ├── MildDemented/
│   ├── ModerateDemented/
│   ├── NonDemented/
│   └── VeryMildDemented/

```
Link for dataset (https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset)


## Model and Training


- Model used: VGG16 (pre-trained on ImageNet)  
- Layers: All convolutional blocks from VGG16 retained; custom dense layers added at the top
- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Image size: 176x208
- Batch size: 20
- Epochs: 20

### Layers Unlocked
Certain convolution layers are set as trainable to fine-tune the network with domain-specific features.



##  Performance Metrics

The model is evaluated using:
- Accuracy (Train / Validation / Test)
- Confusion Matrix
- Precision, Recall, F1-score (for each class)
- ROC Curves and AUC Scores

Sample result:
```
Train Accuracy: 99.84%
Validation Accuracy: 99.68%
Test Accuracy: 98.75%
```

## Visualizations

The following plots are included:
- Sample predictions
- Confusion Matrix
- Precision/Recall/F1 for each class
- ROC Curves with AUC

## Prediction on New Image

To test the model on a new image:
```python
from tensorflow.keras.preprocessing import image

img_path = "test/ModerateDemented/27.jpg"
img = image.load_img(img_path, target_size=(176, 208))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

prediction = vg_model.predict(img_array)
predicted_class = np.argmax(prediction)
print("Predicted class:", CLASSES[predicted_class])
```


##  Web App (Streamlit)

This project includes a simple Streamlit web app (`app.py`) for interactive predictions.

###  Requirements
Install dependencies:
```bash
pip install tensorflow keras numpy matplotlib seaborn scikit-learn pillow streamlit
```

### ▶ Running the App
To launch the Streamlit web app:
```bash
streamlit run app.py
```

The app lets you upload a brain MRI image and instantly predicts the stage of Alzheimer’s disease.



##  Project Structure

```
├── train/
│   ├── MildDemented/
│   ├── ModerateDemented/
│   ├── VeryMildDemented/
│   └── NonDemented/
├── test/
│   └── same structure as train/
├── app.py
├── VGG.hdf5
├── Alzheimers_detection.ipynb
└── README.md
```


## Conclusion

This project demonstrates the effective use of transfer learning (VGG16) for medical image classification. With a well-preprocessed dataset and fine-tuned model, high accuracy is achieved. The Streamlit app adds an interactive layer, making it accessible for real-world testing.




