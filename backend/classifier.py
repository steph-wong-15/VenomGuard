from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from numpy import expand_dims

def classifyImage(file, model):
    # Loads the image and transforms it to (224, 224, 3) shape
    original_image = Image.open(file)
    original_image = original_image.convert('RGB')
    original_image = original_image.resize((224, 224), Image.NEAREST)
    
    numpy_image = image.img_to_array(original_image)
    image_tensor = expand_dims(numpy_image, axis=0)  # Add batch dimension

    processed_image = preprocess_input(image_tensor, mode='caffe')
    preds = model.predict(processed_image)

    # Decode predictions to get class descriptions
    prediction = decode_predictions(preds, top=1)
    # Extract class description of the predicted dog breed
    result = prediction[0][0][1]

    return result
