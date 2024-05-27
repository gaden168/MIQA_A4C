# MIQA_A4C
Cardiac Ultrasound Image Quality Assessment Model

## Requirements

This is my experiment eviroument

- python3.6

- pytorch1.6.0+cu101

- tensorboard 2.2.2(optional)

  ## Usage

### 1. enter directory

```python
$ cd MIQA_A4C
```

### 2. dataset

The data consists of ultrasound A4C images with quality labels. You can refer to the train.json, val.json, and test.json files in the images folder to divide the data into training, validation, and test sets.

### 3. the model

Models prefixed with "BL" perform direct regression of quality scores. For example, the BL_train model takes an image as input and outputs the corresponding quality score through a regression model.

Models prefixed with "ML" are quality assessment models within a multi-task framework. They evaluate image quality comprehensively from both pixel-level and semantic-level assessments.

### 4. train the model

Train the model using train.py

```python
$ python ML_train.py
```

### 5. test the model

Test the model using test.py

```python
$ python ML_test.py
```
