# DSVDD-CPM
Anomaly Detection Control through Zero-Shot Learning in Defect 
Inspection Systems

![dsvdd-cpm](https://github.com/user-attachments/assets/54ddc0a7-ac36-4398-a8a5-30753f231d16)
     
Journal of Institute of Control, Robotics and Systems (202x) 3x(x)      
ISSN:1976-5622       
eISSN:2233-4335

## Usage
### Datasets
Our research currently has completed AD experiments on the MNIST ([http://yann.lecun.com/exdb/mnist/](https://yann.lecun.com/exdb/mnist/)), CIFAR-10 () and MVTec-AD () datasets.



### SWaT data processing
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
