# Animal Tracker Project

This project aims at building and deploying an application to classify animal observation pictures (*e.g.* pictures of animal, tracks, scats, etc.) for education purposes.

The training data is a subset of the Amazon Web Services (AWS) open dataset from [iNaturalist](https://github.com/inaturalist/inaturalist-open-data) containing tens of millions of wildlife observations.

Transfer learning was used to improve prediction performance. A pretrained EfficientNet model was fine tuned on cloud computing platform [Kaggle](http://kaggle.com) to enable GPU acceleration to make training faster.

Model was uploaded on a Google Cloud Platform (GCP) Storage bucket.

Application was deployed on GCP using the Cloud Functions API.

## Usage

For now, app can only be used by sending a HTTP post request containing a the image to be classified (named `file`) to this endpoint `https://us-central1-animal-tracker-418112.cloudfunctions.net/predict`. GCP Cloud Functions returns a dictionary containing the predicted class as well as the confidence value.

## Structure
- `data`: data used for modeling (too large to be hosted here)
- `models`: models and model weights
    - `model`: best model, used in production.
    - `weights`: weights of all the trained models.
- `notebooks`
    - `select_pictures.ipynb`: selects subset of pictures (mammals from eastern Canada and northeastern US).
        - `observations_roi_mammals.parquet`
        - `photo_roi_mammals.parquet`
    - `download_pictures.ipynb`: downloads subset of pictures from AWS S3 bucket.
    - `sort_pictures.ipynb`: sorts pictures and store them into directories by species name (for later import as TensorFlow datasets).
    - `train_model.ipynb`: contains the models training code (this notebook was run on Kaggle.com to benefit from GPU acceleration).
- `src`
    - `tests`: unit tests.
    - `utils`: various modules.
    - `API.py`: Python script for a test FastAPI deployment.


## Dependencies

## Data
Metadata files from AWS S3 bucket, not stored here (around 40Gb total): 
- `observations.csv`
- `photos.csv`
- `taxa.csv`
<!--- - List of dependencies 

## Installation Instructions

Instructions on how to install and set up the project.

## Usage

Explain how to use your project. Provide examples.

## Data

Information about the data used in the project.

## Model Training (if applicable)

Details on how to train or retrain models.

## Evaluation and Results

Metrics used for evaluation and how to interpret results.

## Contributing

Guidelines on how others can contribute to the project.

## License

Specify the license for your project.

## Contact Information

Your contact details.

## Acknowledgments

Any acknowledgments, external contributions, or resources.

## References

References to datasets, papers, or other resources.

## Version History

Keep a record of changes and updates.
-->