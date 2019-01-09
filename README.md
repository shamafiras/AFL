# Adversarial Feedback Loop
[<b>[ArXiv]</b>](https://arxiv.org/pdf/1811.08126)  [<b>[project page]</b>](https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/AFL/)

<b>Official repository of Adversarial Feedback Loop paper<br>
Firas Shama, [Roey Mechrez](http://cgm.technion.ac.il/people/Roey/), Alon Shoshan, [Lihi Zelnik-Manor](http://lihi.eew.technion.ac.il/)<br>
[Technion - Israel Institute of Technology](http://cgm.technion.ac.il/)</b>

<div align='center'>
<img src="readme/abstract.png" width="600px"/><br>
<b>The Adversarial Feedback Loop</b>
</div>


### Improves quality of GANs (CIFAR10):
<div align='center'>
<img src="readme/cifar10.png" width="600px"/>
</div>

### Improves face generation (CelebA):

<img src="readme/faces.gif" width="1058px"/>

### Improves super-resolution:



| <img src="readme/1Guppy_2_loop_0_box.png" width="496px"/> |<img src="readme/1Guppy_2_loop_3_box.png" width="496px"/><br>|
|:---:|:---:|
|<b>Baseline</b> | <b>Ours</b>|



| <img src="readme/1_Guppy_2_HR_croped_crop.png" width="81px"/> | <img src="readme/1_Guppy_2_loop_0_crop.png" width="81px"/> | <img src="readme/1_Guppy_2_loop_1_crop.png" width="81px"/> | <img src="readme/1_Guppy_2_loop_2_crop.png" width="81px"/> | <img src="readme/1_Guppy_2_loop_3_crop.png" width="81px"/> |
| :----------: | :----------: | :----------: | :----------: | :----------: |
| <b>GT</b> | <b>Baseline</b> | <b>iter1</b> | <b>iter2</b> | <b>iter3</b> |



### Code usage
The repository includes the source code used for the paper.
Each application has its own usage guidelines, please follow the instructions in each application directory.
### Citation

If you find our research or code useful for you, please cite our paper:

```
@article{shama2018adversarial,
  title={Adversarial Feedback Loop},
  author={Shama, Firas and Mechrez, Roey and Shoshan, Alon and Zelnik-Manor, Lihi},
  journal={arXiv preprint arXiv:1811.08126},
  year={2018}
}
```

