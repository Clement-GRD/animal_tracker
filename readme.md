# Animal Tracker Project

This project aims at building an application to classify animal observation pictures (*e.g.* pictures of animal, tracks, scats, etc.) for education purposes.

We will base our model on the AWS open dataset from [iNaturalist](https://github.com/inaturalist/inaturalist-open-data) containing tens of millions of wildlife observations.

Our model will be trained on Kaggle to enable GPU acceleration.

## Structure

- `select_pictures.ipynb`: Selects subset of pictures (mammals from eastern Canada and northeastern US).
    - `observations_roi_mammals.parquet`
    - `photo_roi_mammals.parquet`

- `download_pictures.ipynb`: Downloads subset of pictures from AWS S3 bucket.
- `sort_pictures.ipynb`: Sorts pictures and store them into directories by species name (for later import as TensorFlow datasets).
- `train_models.ipynb`: contains the models training code (this notebook was run on Kaggle.com to benefit from GPU acceleration).

## Dependencies

## Data
Metadata files from AWS S3 bucket (around 40Gb total): 
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