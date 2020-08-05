# Preparing Data

A CLI has been built for easy dataset downloading, processing, and transferring.
If not otherwise specified, users can use the following command line.

Example `mnist`
```
python3 proc.py --dataset mnist
```

Supported datasets now include:

1. mnist
2. face_detection
3. imagenet_1k
4. coco2014
5. cifar100
6. voc2009

# Downloading

All source data paths are hard-coded in the individual dataset processing file (e.g. lwll_dataset_prep/mnist/main.py).

Supported source data format include: .tar, .tar.gz, .tgz, and .zip.

To only download the dataset:

Example `mnist`
```
python3 proc.py --dataset mnist --download
```

# Processing

Each dataset is processed individually. For image_classification and object_detection data, we extract the image and correspondent label from the original data format. 
Labels for image_classification are text or numbers.
Labels for object_detection includes class id and bounding box, where we use a consistent representation of:

```
[x_min, y_min, x_max, y_max]
``` 

To only process a dataset:

Example `mnist`
```
python3 proc.py --dataset mnist --process
```

## Notes

### Bounding Boxes

The face_detection dataset has bounding boxes in the form of ellipses [major_axis_radius, minor_axis_radius, angle, center_x, center_y, 1]

The coco2014 dataset has bounding boxes in the form of [x, y, width, height].

The voc2009 dataset has bounding boxes in the form of [x_max, x_min, y_max, y_min].

### cifar100

The processing of cifar100 does not include untar, as we need to run a separate program to convert the pickle format data to png.
The code is at https://github.com/knjcode/cifar2png.

We insert at line 259 with

```
258:  batch_count = defaultdict(int)
259:  count = 0
```

and Line 261 with

```
260: label_count[label] += 1
261: count += 1
```

We replace Line 267

```
filename = '%04d.png' % label_count[label]
```

with

```
filename = '%05d.png' % count
```

Then we call 

```
cifar2png cifar100 path/to/cifar100png
```

to convert pickle format to png.

Finally, we move the `train/` and `test/` to `lwll_datasets/cifar100/`, and we are ready to process the Cifar100 dataset.

# Transferring
The CLI uploads the processed datasets and labels into s3://lwll-datasets/compressed-datasets/development and the labels to s3://lwll-datasets/labels, respectively.

To only transfer a dataset:
Example `mnist`
```
python3 proc.py --dataset mnist --transfer
```
