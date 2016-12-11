####Accuracy: 0.71

####Data:

* train_birds_lmdb: https://drive.google.com/open?id=0BxZ-INMQnIZLd0dDNHN2dmJ0OWc
* test_birds_lmdb: https://drive.google.com/open?id=0BxZ-INMQnIZLb1pURXVxT1Y1VUE

(All pictures have already been resized to 224х224)

####Wights:

VGG_ILSVRC_16_layers: http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

####Usage:

* ```cd .../caffe```

* Upload data to ```./data/birds/```

* Upload weights and prototxt to ```./models/vgg/```

* Run:  ```./build/tools/caffe train -model ./models/vgg/train_val.prototxt -solver ./models/vgg/solver.prototxt -weights ./models/vgg/VGG_ILSVRC_16_layers.caffemodel```
