# Psychology to Semantic distance toolbox (psy2sem)

This repository includes a set of python scripts for semantic distance(similarity) calculation with large corpora.  The target audience is the calculation of semantic distance between _words, word pairs_, and _texts_ in _psychological experiments_. 

## Dependencies

To run the Python scripts `scripts_allstep.py`:

* [Python](https://www.python.org/downloads/) 3.7 or higher
* [Jieba](https://pypi.org/project/jieba/) 0.42.1
* [gensim](https://www.cnpython.com/pypi/gensim) 4.1.2
* [numpy](https://pypi.org/project/numpy/) 1.20.1
* [pandas](https://pypi.org/project/pandas/)  1.2.1

## Installation

After downloading and installing all dependencies, download this repository using the following command line:

```
$ git clone https://github.com/semantics2psycho/Semantic-Toolbox
```


## How to run the codes

The python script `scripts_allstep_english.py` contains all analysis that includes three aspects of functionality.

### text process

This module includes text cleaning, text length, text segmentation, part of speech marking, word frequency statistics, result file saving functions.

### corpus training

The function of this module is to train the corpus into word vectors, which can be adjusted parameters.

A user is only required to set the path in the following code line:

```
basedir = ''
```

### distance calculation

This module includes the calculation of semantic distances between words, word pairs (semantic relationships), sentences and texts.

## Note

According to the above module, we have designed two functions specifically for psychological experiments——AUT (alternative use task)、FAT (free association task). Users can refer to and write codes suitable for their own experiments.



