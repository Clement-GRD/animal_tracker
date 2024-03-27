import numpy as np
from google.cloud import storage
from tensorflow.keras.models import load_model
import PIL
from PIL import Image, ImageOps

animal_classes = ['alces_alces', 'sciurus_carolinensis', 'ursus_americanus',
       'homo_sapiens', 'odocoileuscd_virginianus', 'peromyscus_maniculatus',
       'erethizon_dorsatum', 'vulpes_vulpes', 'blarina_brevicauda',
       'canis_lycaon', 'castor_canadensis', 'neogale_vison',
       'canis_latrans', 'lontra_canadensis', 'tamias_striatus',
       'canis_familiaris', 'ondatra_zibethicus', 'mephitis_mephitis',
       'tamiasciurus_hudsonicus', 'lynx_rufus', 'procyon_lotor',
       'felis_catus', 'marmota_monax', 'sylvilagus_floridanus',
       'lepus_americanus', 'didelphis_virginiana', 'pekania_pennanti',
       'peromyscus', 'canidae', 'urocyon_cinereoargenteus',
       'rattus_norvegicus', 'microtus_pennsylvanicus', 'martes_americana',
       'carnivora', 'parascalops_breweri', 'eptesicus_fuscus',
       'mustela_richardsonii', 'condylura_cristata', 'rodentia',
       'sciuridae', 'leporidae', 'placentalia', 'vespertilionidae']

model = None

def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """
    Download a blob from a Google Cloud Storage bucket to a local file.

    Parameters:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        source_blob_name (str): The name of the source blob to download.
        destination_file_name (str): The local file path where the blob will be downloaded.

    Returns:
        None
    """    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def predict(request) -> dict[str, float]:
    """
    Predict the class and confidence of an image uploaded via a HTTP request.

    Parameters:
        request (Request): The HTTP request object containing the uploaded image.

    Returns:
        dict[str, float]: A dictionary containing the predicted class and confidence level.
    """    

    # Download and load the model if it's not already loaded
    try: 
       
        global model
        if model == None:
            download_blob(
                'animal-tracker-tf-models',
                'models/pretrained_fine_300_ratio.h5',
                '/tmp/pretrained_fine_300_ratio.h5',
            )
            model = load_model('/tmp/pretrained_fine_300_ratio.h5')

    except Exception as e:
        error_message = f"Error loading model: {str(e)}"
        return {'error': error_message}

    # Get the uploaded image from the request and preprocess it    
    try:
        
        image = request.files['file']
        cropped_image = ImageOps.fit(Image.open(image).convert('RGB'), (300, 300))
        image_array = np.expand_dims(np.array(cropped_image), 0)

    except Exception as e:
        error_message = f"Error loading image: {str(e)}"
        return {'error': error_message}
    
    # Make predictions using the model
    prediction_array = model.predict(image_array)
    predicted_class = animal_classes[np.argmax(prediction_array)]
    confidence = round(np.max(prediction_array) * 100, 1)

    return {'predicted_class': predicted_class, 'confidence': confidence}
    

