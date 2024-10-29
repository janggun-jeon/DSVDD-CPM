# DSVDD-CPM
Anomaly Detection Control through Zero-Shot Learning in Defect 
Inspection Systems

![dsvdd-cpm](https://github.com/user-attachments/assets/54ddc0a7-ac36-4398-a8a5-30753f231d16)

[https://doi.org/10.5302/J.ICROS.202x.2x.xxxx](https://www.dbpia.co.kr/journal/publicationDetail?publicationId=PLCT00002128)

Journal of Institute of Control, Robotics and Systems (202x) 3x(x)      
ISSN:1976-5622       
eISSN:2233-4335


## Usage
### Datasets
Our research currently has completed AD experiments on the MNIST ([yann.lecun](https://yann.lecun.com/exdb/mnist/)), CIFAR-10 ([cs.toronto](https://www.cs.toronto.edu/~kriz/cifar.html)) and MVTec-AD ([mvtec](https://www.mvtec.com/company/research/datasets/mvtec-ad/), [kaggle](https://www.kaggle.com/datasets/thtuan/mvtecad-mvtec-anomaly-detection)) datasets.


#### MVTec-AD example by shell script
Run shell script (log file output: ~/DSVDD-CPM/log)
```
sh mecro.sh mvtecad
```

or default setting (MVTec-AD)
```
sh mecro.sh
```

or with options <p></p>
(dataset | decay coef | linear decay | pretrain | class)
```
sh mecro.sh mvtecad 0.9 True True 0
```


#### MVTec-AD example by python execution
Run python execution (ternimal window output)
```
python ./src/main.py mvtecad mvtecad_LeNet_ELU ./log/mvtecad_test ./data
```

or with options <p></p>
(dataset_name | net_name | xp_path | data_path | seed | device | optimizer_name | lr | n_epochs | lr_milestone | batch_size | weight_decay | decay_coef | linear_decay | pretrain | ae_lr | ae_n_epochs | ae_lr_milestone | ae_batch_size | ae_weight_decay | normal_class | n_jobs_dataloader)
```
python ./src/main.py mvtecad mvtecad_LeNet_ELU ./log/mvtecad_test ./data --seed 1758683904 --device cuda --optimizer_name adam --lr 0.01 --n_epochs 60 --lr_milestone 20 --lr_milestone 50 --batch_size 32 --weight_decay 0.5e-6 --decay_coef 0.9 --linear_decay True --pretrain True --ae_lr 0.01 --ae_n_epochs 75 --ae_lr_milestone 60 --ae_batch_size 32 --ae_weight_decay 0.5e-3 --normal_class 0 --n_jobs_dataloader 0
```

Run python execution (log file output: ~/DSVDD-CPM/log)
```
nohup python ./src/main.py mvtecad mvtecad_LeNet_ELU ./log/mvtecad_test ./data --normal_class 0 --decay_coef 0.9 --linear_decay True ./log/mvtecad_test/0/decay_coef=0.9-linear_decay=True.out 2>&1 &
```

<br>


#### MVTec-AD experiment
![table](https://github.com/user-attachments/assets/47ed6b7a-cf82-4b6d-964c-ea24ae0151dd)

#### CIFAR-10 example by shell script
Run shell script (log file output: ~/DSVDD-CPM/log)
```
sh mecro.sh cifar10
```

or with options <p></p>
(dataset | decay coef | linear decay | pretrain | class)
```
sh mecro.sh cifar10 0.9 True True 0
```

#### CIFAR-10 example by python execution
Run python execution (ternimal window output)
```
python ./src/main.py cifar10 cifar10_LeNet_ELU ./log/cifar10_test ./data
```

or with options <p></p>
(dataset_name | net_name | xp_path | data_path | seed | device | optimizer_name | lr | n_epochs | lr_milestone | batch_size | weight_decay | decay_coef | linear_decay | pretrain | ae_lr | ae_n_epochs | ae_lr_milestone | ae_batch_size | ae_weight_decay | normal_class | n_jobs_dataloader)
```
python ./src/main.py cifar10 cifar10_LeNet_ELU ./log/cifar10_test ./data --seed 1170014347 --device cuda --optimizer_name adam --lr 0.0001 --n_epochs 150 --lr_milestone 120 --batch_size 256 --weight_decay 0.5e-6 --decay_coef 0.9 --linear_decay True --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 280 --ae_batch_size 256 --ae_weight_decay 0.5e-6 --normal_class 0 --n_jobs_dataloader 0
```

Run python execution (log file output: ~/DSVDD-CPM/log)
```
nohup python ./src/main.py cifar10 cifar10_LeNet_ELU ./log/cifar10_test ./data --normal_class 0 --decay_coef 0.9 --linear_decay True ./log/cifar10_test/0/decay_coef=0.9-linear_decay=True.out 2>&1 &
```

<br>

#### CIFAR-10 experiment
![cifar](https://github.com/user-attachments/assets/0c67f7d4-9148-4cc2-9083-c67abc57e064)


#### MNIST example by shell script
Run shell script (log file output: ~/DSVDD-CPM/log)
```
sh mecro.sh mnist
```

or with options
```
sh mecro.sh mnist 0.9 True True 0
```

#### MNIST example by python execution
Run python execution (ternimal window output)
```
python ./src/main.py mnist mnist_LeNet ./log/mnist_test ./data --lr 0.0001 --n_epochs 150 --lr_milestone 120 --batch_size 256 --weight_decay 0.5e-6 --decay_coef 0.9 --linear_decay True --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 120 --ae_batch_size 256 --ae_weight_decay 0.5e-3 --normal_class 0
```

Run python execution (log file output: ~/DSVDD-CPM/log)
```
nohup python ./src/main.py mnist mnist_LeNet ./log/mnist_test ./data --normal_class 0 --decay_coef 0.9 --linear_decay True ./log/mnist_test/0/decay_coef=0.9-linear_decay=True.out 2>&1 &
```


## Reference
If you find this repo helpful to your research, please cite our paper.
```
@article{,
  title={Anomaly Detection Control through Zero-Shot Learning in Defect Inspection Systems},
  author={Janggun Jeon, Namgi Kim},
  journal={Journal of Institute of Control, Robotics and Systems},
  year={2024}
}
```
