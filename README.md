# Anomaly Detection

This repository contains an implementation of an anomaly detection system. The system is designed to identify unusual patterns or outliers in a given dataset. 

## Table of Contents

- [Introduction](#introduction)
- [Approaches](#approaches)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Algorithms](#algorithms)

## Introduction

Anomaly detection plays a critical role in various domains such as fraud detection, intrusion detection, system health monitoring, and many more. This project aims to provide a robust and customizable anomaly detection system that can be applied to different types of datasets.
 
## Approaches
When we get analise the data we all had different approaches to tackle the problem statement some of them are listed below -
1. We thought of treating this problem statement as semi-supervised learning. We would handpick some data which we think are anomly of different kinds and then apply featuren engineering and some algorithms on that data to see which model will give us the desired output . We had to drop that idea cause the dataset was so large we can't open the file on our systems and handpick or observe the data.
2. There were manay different opinions on how could we reduce dimensions and make data more readable to computer. Some of them were data embeding , different encoding methods which we have used.
3. Finally we all agreed on using the K-means and some encoding methods in [Model](https://github.com/tans-hul/Anomly-Detection/blob/main/Model%20Creation%20(10%20lakh)%20(test).ipynb) file.

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
   python Api.py
   ```

## Usage

Once the system is running, you can specify the input dataset and choose the appropriate anomaly detection algorithm. The system will process the data and provide you with the detected anomalies. 

For detailed usage instructions and examples, please refer to the [documentation](#).

## Dataset

The repository includes a sample dataset located in the `data` directory. You can replace it with your own dataset or use the provided data as a starting point for experimentation. Make sure your dataset is properly formatted and compatible with the selected anomaly detection algorithm.

## Algorithms

The repository currently supports the following anomaly detection algorithms:

- [K-means](https://en.wikipedia.org/wiki/K-means_clustering)

We tried and tested different algorithms such as Hierarchical Clustering, DBSCAN, Spectral Clustering, GMM and K Means Clustering but k-means gave the efficient result according to our observation.

