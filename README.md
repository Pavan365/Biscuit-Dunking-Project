# Machine Learning - Biscuit Dunking Project

## Contents
1. [Overview](#overview)
2. [Usage](#usage)
    + [Setup](#Setup)
    + [Workflow](#workflow)
3. [Codebase Structure](#codebase-structure)

## Overview
<p align=justify>
Welcome! This is the codebase for the machine learning project of investigating 
data related to the structural properties of biscuits when dunked in tea.
</p>

## Usage
<p align=justify>
This section provides a quick overview on how to download the codebase and go 
through the workflow performed in this project.
</p>

### Setup
+ The codebase can be cloned using the following command.
```
$ git clone git@github.com:Pavan365/Biscuit-Dunking-Project.git
```

+ Change directories into the codebase, install and activate the conda environment.
```
$ cd Biscuit-Dunking-Project
$ conda env create -f environment.yaml
$ conda activate biscuits
```

> [!IMPORTANT]
> Ensure to select the correct environment when running the ***Jupyter Notebooks***.

### Workflow
<p align=justify>
The project first starts with exploring the datasets. The workflow and code for 
this part can be found in the <b><i>exploration</i></b> notebook.
</p>

```
PATH: ./src/data-exploration/exploration.ipynb
```

<p align=justify>
The project then moves onto investigating the classification of biscuits using 
a machine learning model. This involves comparing SVC, RFC and NNC models, and 
performing classification on the <b><i>microscope</i></b> data. The workflow 
and code for this part can be found in the <b><i>classification</i></b> 
notebook.
</p>

```
PATH: ./src/data-classification/classification.ipynb
```

<p align=justify>
The project finalises with investigating the modelling and prediction of tea 
flow up each biscuit. This first section involves comparing the performance of 
the Washburn equation and a RFR model on the <b><i>microscope</i></b> data. The 
second section involves investigating the performance of the Washburn equation 
on the <b><i>time series</i></b> data and improving it with a modified Washburn 
model. The workflow and code for this part can be found in the 
<b><i>modelling</i></b> notebook.
</p>

```
PATH: ./src/data-modelling/modelling.ipynb
```

## Codebase Structure
```
.
├── README.md
├── environment.yaml
└── src
    ├── data-classification
    │   ├── biscuit_classification.ipynb
    │   └── classifier.py
    ├── data-exploration
    │   └── exploration.ipynb
    ├── data-modelling
    │   └── flow_modelling.ipynb
    └── datasets
        ├── dunking-data
        │   └── dunking.csv
        ├── microscope-data
        │   ├── microscope-classified.csv
        │   └── microscope.csv
        └── time-series-data
            ├── time-series-1.csv
            ├── time-series-2.csv
            └── time-series-3.csv
```

#### ```./README.md```
+ This ***README*** file.

#### ```./environment.yaml```
+ The conda environment used for this project.

#### ```./src/data-classification/```
+ Contains the notebook and script used to perform data classification.

#### ```./src/data-exploration/```
+ Contains the notebook used to perform initial data exploration.

#### ```./src/data-modelling/```
+ Contains the notebook used to perform data modelling and regression.

#### ```./src/datasets/```
+ Contains the datasets (CSV files) investigated in this project.