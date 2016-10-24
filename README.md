# Neural-Emotion

Emotion prediction and application with neural networks.

## Dependencies

* dlib
* scikit image
* nolearn
* theano
* lasagne
* matplotlib
* numpy, scipy stack

To install:

```BASH
cd Neural-Emotion
sh dependencies.sh
```

## Performance

```
# Neural Network with 16178088 learnable parameters

## Layer information

name       size         total    cap.Y    cap.X    cov.Y    cov.X
---------  ---------  -------  -------  -------  -------  -------
input      1x100x100    10000   100.00   100.00   100.00   100.00
conv2d1    64x96x96    589824   100.00   100.00     5.00     5.00
maxpool1   64x48x48    147456   100.00   100.00     5.00     5.00
conv2d2    32x44x44     61952    76.92    76.92    13.00    13.00
maxpool2   32x22x22     15488    76.92    76.92    13.00    13.00
dropout1   32x22x22     15488   100.00   100.00   100.00   100.00
dense      1024          1024   100.00   100.00   100.00   100.00
dropout2   1024          1024   100.00   100.00   100.00   100.00
dense1     256            256   100.00   100.00   100.00   100.00
gaussian1  256            256   100.00   100.00   100.00   100.00
output     8                8   100.00   100.00   100.00   100.00

Explanation
    X, Y:    image dimensions
    cap.:    learning capacity
    cov.:    coverage of image
    magenta: capacity too low (<1/6)
    cyan:    image coverage too high (>100%)
    red:     capacity too low and coverage too high


  epoch    trn loss    val loss    trn/val    valid acc  dur
-------  ----------  ----------  ---------  -----------  -----
...
    227     4.49510     2.69445    1.66828      0.71967  0.70s
    228     4.23806     2.69361    1.57338      0.71967  0.71s
    229     4.16328     2.72581    1.52735      0.71967  0.68s
    230     4.25037     2.73666    1.55312      0.71967  0.68s
    231     4.43495     2.72948    1.62483      0.71967  0.71s
    232     4.60830     2.70795    1.70176      0.71967  0.76s
    233     4.13195     2.68228    1.54046      0.71967  0.76s
    234     4.17592     2.63753    1.58327      0.71967  0.71s
    235     4.11365     2.61233    1.57471      0.71967  0.68s
    236     4.35378     2.59228    1.67952      0.71967  0.68s
    237     4.46442     2.58754    1.72535      0.71967  0.73s
    238     4.30135     2.58465    1.66419      0.71967  0.74s
```