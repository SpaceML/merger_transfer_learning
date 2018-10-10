# Transfer Learning

This project is the implementation of the learning protocol in the paper _"Using transfer learning to detect galaxy mergers"_. This code can be used to exactly replicate the results of the paper with the original merger dataset, or for training with a completely new dataset.

## Prerequisites
Linux or OSX

NVIDIA GPU + CUDA CuDNN for training on GPU's

## Dependencies
The code is tested on <tt>python 3.5</tt>

Training requires the following python packages: <tt>tensorflow</tt>, <tt>numpy</tt>, <tt>keras</tt>, <tt>tqdm</tt>

## Get our code
Clone the repo:
```bash
git clone https://github.com/SpaceML/merger_transfer_learning.git
cd  merger_transfer_learning/
```

## Run our code

### Directory structure for dataset
The original dataset can be found in this Google Drive folder:
```https://drive.google.com/open?id=1klSL9wZs3cZqAqmvFJVHxkPyXCbXN0-j```

The code assumes a directory structure like the following one:

```bash
dataset_root/
└── merger
    └── training
        └── images
            ├── 00001.jpeg
             ⋮
    └── validation
        └── images
            ├── 01001.jpeg
             ⋮
    └── test
        └── images
            ├── 02001.jpeg
             ⋮
└── noninteracting
    └── training
        └── images
            ├── 00001.jpeg
             ⋮
    └── validation
        └── images
            ├── 01001.jpeg
             ⋮
    └── test
        └── images
            ├── 02001.jpeg
             ⋮
```

If you wish to use different class label names, please modify the respective default arguments in ```make_generators(path_images, positive_class, negative_class)``` in the file `training.py`.

If you use your own data set, please also update the corresponding default arguments in ```training_loop(...)``` in the file `training.py`.


### Training

```bash
python training.py --path_images /path/to/dataset --path_checkpoints /where/checkpoints/go --path_statuses /where/logging/goes --mode [transferlearning|randinit]
```

The training will not stop automatically, you can monitor the progress in terms of the performance metrics on the validation set, and terminate the training manually as soon as overfitting occurs.
