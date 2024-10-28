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

#### MVTec-AD example


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
sh mecro.sh mvtecad 0.9 False True 0
```

<br><br>

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

![table](https://github.com/user-attachments/assets/a5745804-6fff-4af3-adec-38f08a061eab)

# Or
sh mecro.sh
# Run shell script (output: terminal)

1. Run `unzip ./SWaT/data/SWaT.zip` to unzip the datasets      
or      
2. Run `cd ./SWaT/utils`     
   Run `python gdrivedl.py https://drive.google.com/open?id=1rVJ5ry5GG-ZZi5yI4x9lICB8VhErXwCw ./SWaT`      
   Run `python gdrivedl.py https://drive.google.com/open?id=1iDYc0OEmidN712fquOBRFjln90SbpaE7 ./SWaT`      
   Run `mkdir -p ./../data`      
   Run `mv ./SWaT ./../data/SWaT`     

### Traing & Evaluation
#### SMD datasets
`SMD`      
`machine-1-1`, `machine-1-2`, `machine-1-3`, `machine-1-4`, `machine-1-5`, `machine-1-6`, `machine-1-7`, `machine-1-8`,      
`machine-2-1`, `machine-2-2`, `machine-2-3`, `machine-2-4`, `machine-2-5`, `machine-2-6`, `machine-2-7`, `machine-2-8`, `machine-2-9`,      
`machine-3-1`, `machine-3-2`, `machine-3-3`, `machine-3-4`, `machine-3-5`, `machine-3-6`, `machine-3-7`, `machine-3-8`, `machine-3-9`,      
`machine-3-10`, `machine-3-11`      

#### to run of `SMAP`, `MSL` and `SMD` datasets
1. Run `main.ipynb` by jupyter      
or    
2. Run main.py by python 
```
# available models : IF, USAD, Encoded-IF
python main.py --dataset SMAP 
python main.py --dataset MSL 
python main.py --dataset SMD

# available sub-SMD datasets
# python main.py --dataset machine-{a}-{b} --model Encoded-IF --max_epoch 0
# a = {1, 2, 3}
# b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
```

#### to run of `SWaT` datasets
1. Run `/SWaT/IsolationForest.ipynb` by jupyter
2. Run `/SWaT/AutoEncoder.ipynb` by jupyter
3. Run `/SWaT/USAD.ipynb` by jupyter
4. Run `/SWaT/Encoded-IF.ipynb` by jupyter

## Data description
|Dataset|Train|Test|Dimensions|
|:----|:----|:----|:----|
|SWaT|496,800|449,919|51|
|SMAP|135,183|427,617|25|
|MSL|58,317|73,729|55|
|SMD|708,405|708,420|28*28|


