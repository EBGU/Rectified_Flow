# Rectified Flow
A replication of [Rectified Flow](https://arxiv.org/abs/2209.03003) paper with PyTorch and [U-ViT](https://arxiv.org/pdf/2209.12152.pdf).

<p align="center" style="margin-bottom: 0px;">
  <img src="images/01.png" width="100" />
  <img src="images/02.png" width="100" /> 
  <img src="images/03.png" width="100" />
  <img src="images/04.png" width="100" />
</p>

<p align="center" style="margin-bottom: 0px;">
  <img src="images/05.png" width="100" />
  <img src="images/06.png" width="100" /> 
  <img src="images/07.png" width="100" />
  <img src="images/08.png" width="100" />
</p>

<p align="center">
  <img src="images/09.png" width="100" />
  <img src="images/10.png" width="100" /> 
  <img src="images/11.png" width="100" />
  <img src="images/12.png" width="100" />
</p>
To train a new model, you can modify the yaml file and:

` python multi_gpu_trainer.py example `

Training data of [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/) should be split manually, and you can find the numpy version of their labels in this repo.

To run inference, please  download my pretrained weight(will be provided later):

` python sample_img.py --device "cuda:0" --load "last" --SavedDir tmp/ --ExpConfig example/example.yaml --n_sqrt 16 --steps 200 `

The inference process is controled by 6 parameters :

"device", usually 'cuda:0' ;

"load", best epoch or last epoch;

"SavedDir", where to save images;

"ExpConfig", the yaml file of your experiments;

"n_sqrt", you will get N<sup>2</sup> samples for each class;

"steps", n steps for sampling, in my experiment, 200 is a good choice.

The result should looks like the welcoming images.

Enjoy!

