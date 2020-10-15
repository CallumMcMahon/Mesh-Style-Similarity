# A fork of MeshCNN in PyTorch

This repository provides supporting code for my Masters thesis: [On Stylistic Differences Between 3D Meshes](On%20Stylistic%20Differences%20Between%203D%20Meshes.pdf)

This fork extends the original [meshCNN](https://github.com/ranahanocka/MeshCNN) implementation by [Rana Hanocka](https://www.cs.tau.ac.il/~hanocka/) and [Amir Hertz](http://pxcm.org/) for their SIGGRAPH 2019 [paper](https://bit.ly/meshcnn)

MeshCNN is a general-purpose deep neural network for 3D triangular meshes, which can be used for tasks such as 3D shape classification or segmentation. This framework includes convolution, pooling and unpooling layers which are applied directly on the mesh edges.

Extensions are as follows:
 - optimised pooling operation for linear space complexity instead of quadratic, greatly increasing the sizes of meshes able to be processed
 - support for meshes composed of many smaller components, where pooling of small components to single faces does not break the model
 - implemented edge colouring for obj file exports for viewing in external programs such as MeshLab
 - changed much of the boilerplate code for model training to [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
 - examples of back-propagated gradients to the original mesh to visualise edges/regions which contributed to classification
 - gram-based style loss for identification of stylistic differences between meshes, visualising regions of dissimilarity
 - example pre-processing scripts to attempt to deal with low-quality meshes using [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus)
 - made the code-base slightly easier to navigate with function description comments, simplifying pytorch objects like datasets & dataloaders, and removal of empty files



# Getting Started

### Installation
- Clone this repo:
```bash
git clone https://github.com/ranahanocka/MeshCNN.git
cd MeshCNN
```
- Install dependencies: [PyTorch](https://pytorch.org/), <i> Optional </i>: [tensorboardX](https://github.com/lanpa/tensorboardX) for training plots, [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training.
  
### Datasets
For the original datasets used in the meshCNN paper, please refer to the [main repo](https://github.com/ranahanocka/MeshCNN)

Style-based analysis uses the Co-Locating Style-Defining Elements on 3D Shapes dataset found [here](http://vcc.szu.edu.cn/research/2017/style/).
<!---
```bash
bash ./scripts/shrec/get_data.sh
```

Run training (if using conda env first activate env e.g. ```source activate meshcnn```)
```bash
bash ./scripts/shrec/train.sh
```

To view the training loss plots, in another terminal run ```tensorboard --logdir runs``` and click [http://localhost:6006](http://localhost:6006).

Run test and export the intermediate pooled meshes:
```bash
bash ./scripts/shrec/test.sh
```

Visualize the network-learned edge collapses:
```bash
bash ./scripts/shrec/view.sh
```


An example of collapses for a mesh:

<img src="/docs/imgs/T252.png" width="450px"/> 

Note, you can also get pre-trained weights using bash ```./scripts/shrec/get_pretrained.sh```. 

In order to use the pre-trained weights, run ```train.sh``` which will compute and save the mean / standard deviation of the training data. 


### 3D Shape Segmentation on Humans
The same as above, to download the dataset / run train / get pretrained / run test / view
```bash
bash ./scripts/human_seg/get_data.sh
bash ./scripts/human_seg/train.sh
bash ./scripts/human_seg/get_pretrained.sh
bash ./scripts/human_seg/test.sh
bash ./scripts/human_seg/view.sh
```

Some segmentation result examples:

<img src="/docs/imgs/shrec__10_0.png" height="150px"/> <img src="/docs/imgs/shrec__14_0.png" height="150px"/> <img src="/docs/imgs/shrec__2_0.png" height="150px"/> 

### Additional Datasets
The same scripts also exist for COSEG segmentation in ```scripts/coseg_seg``` and cubes classification in ```scripts/cubes```. 

# More Info
Check out the [MeshCNN wiki](https://github.com/ranahanocka/MeshCNN/wiki) for more details. Specifically, see info on [segmentation](https://github.com/ranahanocka/MeshCNN/wiki/Segmentation) and [data processing](https://github.com/ranahanocka/MeshCNN/wiki/Data-Processing).
--->

# Modified Pooling
The original implementation method stores an n x n dense matrix for saving which edge features are averaged during the pooling phase. This matrix ends up being very sparse, only storing a few values per row depending on how many pooling operations are needed. This can be replaced with a [Pytorch sparce matrix (still in beta)](https://pytorch.org/docs/stable/sparse.html) which enabled the use of much larger meshes, experimentally over an order of magnitude larger. This removed the need for hacks like splitting the mesh into parts to be passed through the network before reassembly, as is seen in the [point2mesh repo](https://github.com/ranahanocka/point2mesh). However due to sparce matrices being in beta, efficient slicing is not implemented. A crude implementation is provided however this greatly reduces performance by 8x to 10x. The change is given in the [mesh union](models/layers/mesh_union.py) file, and the call to meshUnion in [mesh pool](models/layers/mesh_pool.py) should be changed to MeshUnionSparse to use the sparce implementation. 

Pooling also does not break when sub-components are pooled to a single face. This code is modified to skip this face, and continue to pool the rest of the mesh. If the mesh cannot be pooled the necessary amount due to many unpoolable components, the mesh is left as-is and other meshes in the batch are padded to accomodate.

# External libraries
The design philosophy of the main reposotory aims to keep all requirements included with the project. While this can be useful, added functionality was chosen for this fork. Pytorch-lightning code replaces the functionality of the original [mesh_classifier file](https://github.com/ranahanocka/MeshCNN/blob/master/models/mesh_classifier.py).

A method for saving edge colourings to .obj files is implemented. Since obj files support node colourings and not edge colourings, new faces and nodes are added along each edge with the appropriate colouring. This means they can be interpreted by mesh renderers such as MeshLab for viewing. This addition can be seen from the [mesh.py file](models/layers/mesh.py).

# Citation
If the original model is all that is used then the original paper should be cited, details on the [main repo](https://github.com/ranahanocka/MeshCNN)


# Questions / Issues
If you have questions or issues running this code, please open an issue so we can know to fix it.
