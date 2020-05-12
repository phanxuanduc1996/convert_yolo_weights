With Yolo-v2.
Please replace this code by the code below.

[reorg]
stride=2

[route]
layers=-1,-4


Convert ---->>>>


[maxpool]
size=2
stride=2

[route]
layers=-2

[maxpool]
size=2
stride=2

[route]
layers=-4

[maxpool]
size=2
stride=2

[route]
layers=-6

[maxpool]
size=2
stride=2

[route]
layers=-1,-3,-5,-7

[route]
layers=-1, -11
