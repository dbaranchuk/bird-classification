Точность: 0.71

Данные:

train_birds_lmdb: https://drive.google.com/open?id=0BxZ-INMQnIZLd0dDNHN2dmJ0OWc
test_birds_lmdb: https://drive.google.com/open?id=0BxZ-INMQnIZLb1pURXVxT1Y1VUE

(Все изображения уже обрезаны и приведены к размеру 224х224)

Веса:

VGG_ILSVRC_16_layers: http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel


```cd .../caffe```

Загружаем данные в ```./data/birds/```

Загружаем веса и prototxt файлы в ```./models/vgg/```

Запускаем:  ```./build/tools/caffe train -model ./models/vgg/train_val.prototxt -solver ./models/vgg/solver.prototxt -weights ./models/vgg/VGG_ILSVRC_16_layers.caffemodel```
