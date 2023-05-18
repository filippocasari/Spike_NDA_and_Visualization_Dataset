# Training NDA + VGG11
@authors: Filippo Casari, Alessandro De Grandi, Anthony Bugatto\
Paper reference: https://arxiv.org/pdf/2203.06145.pdf \
First, you need to preprocessing the dataset with **spikingjelly**. Run this script:
```bash
python preprocessing_spikingjelly.py
```
For training NDA and VG11 please refer to this Readme file: [NDA_SNN/README.md]()
We added some arguments to pass to the main.py script in NDA_SNN folder. Now you can pass the **pretrained model** path. If you pass it, you should pass also the the correct starting epoch.  
Additionally, we add the possibility to monit the training with **Wandb**. \
We changed the main.py also to save the model every epoch. Furthermore, to make it suitable for working with **mps** backend instead of **cuda**, we changed slightly the code. \ 
_Note_: don't forget to add --nda option which performs the training with Neuromorphic Data Augmentation. 
Models are saved in the folder: **NDA_SNN/models**. 
## For training :
```bash
cd NDA_SNN
python main.py --dset nc101 --amp --nda
```
The authors accomplished their maximum accuracy after 200 epochs, however, we noticed that just after 45 epochs the model reaches above 70 % of accuracy. 

# For visualizing the datasets:
Datasets must be in folder data/extract. 
## Plot Ncars: 
Be careful where you downloaded Ncars dataset. 
```bash
PYTHONPATH=$(pwd) AEGNN_DATA_DIR=$(pwd) AEGNN_LOG_DIR=$(pwd)/data/log python plotting_ncars.py --dataset "ncars"
```
The program will ask you to choose between visualing the event histogram or the graph representation. The plotted samples are 100 by default, but you can change it within the code. \
**_Note_**: make sure you have aegnn repo with all dependencies installed. Our script uses some functions written in aegnn repo.
## Plot events of Ncaltech101:
tonic library at: https://github.com/neuromorphs/tonic.git \
Thanks to **tonic** library, we created this simple script for plotting events as grid of frames and as videos/animations. \
Like before, you can choose a class from the dataset you prefer most. 
To run the script:
```bash
python plot_events_ncaltech.py
```
