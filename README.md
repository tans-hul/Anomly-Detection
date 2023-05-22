# Anomaly Detection

This repository hosts an implementation of an anomaly detection system. Its primary objective is to proficiently recognize uncommon patterns or outliers within a given dataset.

## Table of Contents

- [Introduction](#introduction)
- [Approaches](#approaches)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Algorithms](#algorithms)

## Introduction

Anomaly detection assumes a pivotal role across diverse domains, including fraud detection, intrusion detection, system health monitoring, and more. The objective of this project is to offer a resilient and adaptable anomaly detection system capable of addressing various types of datasets.
 
## Approaches
During the data analysis phase, we adopted diverse approaches to address the problem statement. Some of these approaches included:
1. Initially, we considered a semi-supervised learning approach but dropped it due to the dataset's size limitations for manual data selection.
2. Various techniques, such as PCA (Principal Component Analysis) and different encoding methods, were explored to reduce dimensionality and enhance the interpretability of the data.
3. Finally, we agreed to use K-means, GMM, and specific encoding methods in our [model](https://github.com/tans-hul/Anomly-Detection/blob/main/notebooks/Model%20Creation%20(1%20crore)%20(main).ipynb).

## Installation

To use this repository, please follow the steps below:

1. Clone the repository:

   ```bash
   git clone https://github.com/tans-hul/Anomly-Detection.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the anomaly detection system:

   ```bash
   python API.py
   ```

## Usage

Once the system is operational, you have the flexibility to specify the input dataset and select the suitable anomaly detection algorithm. The system will then process the data and present you with the identified anomalies.

For detailed usage instructions and examples, please refer to the [documentation](#).

## Dataset

The repository includes a sample dataset located in the `data` variable. You can replace it with your own dataset or use the provided data as a starting point for experimentation. Make sure your dataset is properly formatted and compatible with the selected anomaly detection algorithm.

## Algorithms

The repository currently supports the following anomaly detection algorithms:

- [K-means](https://en.wikipedia.org/wiki/K-means_clustering)
- [GMM](https://en.wikipedia.org/wiki/Mixture_model)

We experimented with several algorithms, including Hierarchical Clustering, DBSCAN, Spectral Clustering, GMM, and K-means Clustering. However, based on our observations, K-means and GMM yielded the most efficient results.

