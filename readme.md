- cfg:
- checkpoint:
- dataloader:
- init_weights:
- modeling:
    + backbone:
        * res_net.py
        * vgg.py
    + classification:
        * google_net.py
        * inception_v2.py
        * inception_v3.py
        * inception_v4.py
        * lenet5.py
    + detection:
        * yolo_v1.py
        * yolo_v2.py
        * yolo_v3.py
    + segmentation:
        * alex_net.py
        * deeplab_v1.py
        * fcn.py
- utils:


