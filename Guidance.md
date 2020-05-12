# Recommendation System Based on User-Book Interactions
DS-GA1004 Final Project Report

## Introduction
In this project, we used Apache Spark ML to develop and evaluate a book recommendation system with implicitfeedback modeling. Alternating Least Square (ALS), a matrix factorization algorithm, is applied to generate theuser and item embedding matrices.  The data that we used is from [Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home). For this project, we used the rating scores as the user-book interactions to train the model and obtain all the evaluation results.

## Data
On Dumbo's HDFS, you will find the datasets after subsampling and splitting in `hdfs:/user/ss13289`:
* `train1.parquet`
* `valid1.parquet`
* `test1.parquet`

The generated latent factors files and book genres file can be found [here](https://drive.google.com/open?id=1e547kWwQtJcHfySQrH_Glq8Rz2s-iG2Z).

## Instructions
* Installation

To install the [annoy](https://github.com/spotify/annoy) package, simply do `pip install --user annoy` to pull down the latest version from [PyPI](https://pypi.org/project/annoy/).


* How to run the `.py` files on the <b>Dumbo cluster</b>:
```
$ module load python/gnu/3.6.5
$ module load spark/2.4.0
$ spark-submit evaluation_script.py
```

## Files
* `evaluation_script.py`  

The full python codes that can be run in PySpark, including codes on baseline model and hyperparameter tuning, and hwo we use the best model (`rank=20, regParam=1, alpha=1`) to evaluate on the test set and get latent facotrs for both users and items

* `Extension_1_Fast_Search.ipynb`  

Codes for Extension 1: Using the annoy package to accelerate the query search

* `Extension_2_Visualization.ipynb` 

Codes for Extension 2: The Visualization of items based on learned item factors from ALS and genres tags using T-SNE

* `dumbo_try_evaluation.py`, `local_try_subset_model.ipynb`, `local_try_subset_split.ipynb`:  

Other codes we have used both locally and on dumbo for trial and subset implementation

* `parameters_tune.xlsx`  

The results of hyperparameter tuning

* `1004_Final_Report.pdf`

The final report




## Team Members

* Anqi Zhang (az983) 
* Yuan Ding (yd1400)
* Shuwen Shen (ss13289)
