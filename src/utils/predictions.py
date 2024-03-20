import numpy as np
import matplotlib.pyplot as plt

def predict(preprocessed_image: np.ndarray, model) -> tuple[str, float]:
    """
    Predicts the class and likelihood of an animal in the given preprocessed image.

    Parameters:
        preprocessed_image (np.ndarray): The preprocessed image array.
        model: TensorFlow model to use for prediction.

    Returns:
        tuple[str, float]: A tuple containing the predicted class name and its likelihood.
    """    
    animal_classes = ['alces_alces', 'sciurus_carolinensis', 'ursus_americanus',
       'homo_sapiens', 'odocoileus_virginianus', 'peromyscus_maniculatus',
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
    
    predictions = model.predict(preprocessed_image)
    predicted_class = animal_classes[np.argmax(predictions)]
    prediction_likelihood = np.round(np.max(predictions)*100, 2)
    
    return predicted_class, prediction_likelihood

def plot_image_prediction(preprocessed_image, model):
    """
    Displays the preprocessed image along with the predicted class and its likelihood.

    This function uses a trained model to predict the class of the given preprocessed image, 
    then plots the image and annotates it with the predicted class and the likelihood 
    of the prediction.

    Parameters:
        preprocessed_image (np.ndarray): The preprocessed image array.
        model (Any): The trained model used for making predictions.

    Returns:
        None
    """
    predicted_class, prediction_likelihood = predict(preprocessed_image, model)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(preprocessed_image[0, :])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    fig.suptitle(predicted_class.replace('_', ' ').title() +f' ({prediction_likelihood}%)', y=0.92)