# Bone Layer State transfomer
BLST is a deep learning project for bone state recoginition, it includes deep learning and machine learing models. specific model implemented in this project are carefully discussed in our paper.


## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtained the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), Then place the downloaded data under the folder `dataset` which is made outside the project.
~~~
    -- project_folder
    -- dataset
~~~
3. Train and evaluate model. We provide the experiment script under the folder `./scripts/`. You can reproduce the experiment results asfollows:

```
bash ./scripts/bone_drill/test_one.sh
```
4. Train ML algorithm like SVM and KNN, directly run the file by jupyter notebook
~~~
bone_drill_ml.ipynb
~~~
## Citation

If you find this repo useful, please cite our paper.

```

```

## Contact
If you have any questions or suggestions, feel free to contact:

- Junhui Huang (jh4165@columbia.edu)

or describe it in Issues. 😊.

## Acknowledgement

This library is constructed based on the following repos:

- TimeNet: https://github.com/thuml/Time-Series-Library
- AutoFormer: https://github.com/thuml/Autoformer

