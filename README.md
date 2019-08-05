# FLARe: Forecasting by Learning Anticipated Representations
<center> Surya Teja Devarakonda*, Yeahuay Joie Wu*, Yi Ren Fung, Madalina Fiterau </center>
<center><italics>University of Massachusetts, Amherst</italics></center>  

*Authors Contributed Equally

#### Machine Learning for Healthcare Conference (MLHC) 2019

## Abstract
Computational models that forecast the progression of Alzheimer's disease at the patient level are extremely useful tools for identifying high risk cohorts for early intervention and treatment planning. The state-of-the-art work in this area proposes models that forecast by using latent representations extracted from the longitudinal data across multiple modalities, including volumetric information extracted from medical scans and demographic info. These models incorporate the time horizon, which is the amount of time between the last recorded visit and the future visit, by directly concatenating a representation of it to the data latent representation. In this paper, we present a model (FLARe) which generates a sequence of latent representations of the patient status across the time horizon, providing more informative modeling of the temporal relationships between the patient's history and future visits. Our proposed model outperforms the baseline in terms of forecasting accuracy and F1 score with the added benefit of robustly handling missing visits. 

Our academic paper which describes FLARe in detail and provides comprehensive results can be found [here](https://arxiv.org/abs/1904.08930).

## Model Description
The objective of FLARe is to predict what the disease stage of the patient could be at a future time point, given his/her medical history data, including MRI scans, cognitive assessments and demographics. The model pipeline is as follows:  
1. **Feature Extraction**: First, we use three seperate multilayer perceptrons (MLPs), one for each category of input features, to encode our input features into a common latent space. We use the manually extracted features for the MRI images, which are provided in the TADPOLE challenge, as inputs for one MLP. We use 4 cognitive scores and 3 demographic values (age, gender and presence of APOE4 gene) as inputs for the other two MLPs. After we extract the representations, we concatenate them. 
2. **Feature Prediction**: We do this for all the available timepoints in the patient's medical history. Then, the sequence of concatenated features is sent to an RNN which provides hidden layer outputs for features of each time point. These hidden layer outputs are fed to another MLP, which performs feature prediction, i.e., it predicts the feature vector of the next time point given the RNN hidden layer output of the present time point. We introduce an auxiliary error loss (typically the mean squared error loss) to optimize this MLP. We continue iteratively generating the sequence of representations for the datapoints between our last available visit and the future visit we want to forecast.  
On the other hand, in our baseline model, which we constructed taking ideas from the state-of-the-art models, we just take the final hidden layer output of the RNN and concatenate a representation of the time difference between the final available time point and the future desired time point. We hypothesize that our approach will be more robust and natural compared to the baseline model.  
3. **Classification**: Finally, we feed the output of the feature prediction module to a MLP classifier, which performs a 3-class classification between the classes Cognitively Normal (CN), Mildly Cognitively Impaired (MCI), and Alzheimer’s Disease (AD).

Baseline model (RNN-Concat):  

![](https://www.dropbox.com/s/noc0v68v6g48ti0/flare_baseline.png?raw=true)

Proposed model (FLARe):

![proposed](https://www.dropbox.com/s/gglrxqgra1n08s4/flare_proposed.png?raw=true)

## Results
To analyze our proposed model’s change in performance across different levels of dataavailability and forecasting horizons, we partition the testing set into buckets where eachbucket corresponds to an ordered pair(T, τ): the number of points used for prediction andthe forecasting horizon. In the following table, we provide the F1 score of RNN-Concat and FLARe for each bucket. 
 
![](https://www.dropbox.com/s/p63j09hey8yaw8i/results.png?raw=true)

### Data
We used data from the [TADPOLE challenge](https://tadpole.grand-challenge.org/Data/#Data). Specifically, we used the **TADPOLE_D1_D2.csv** file. Download 
that file and save it in the directory *data/*.  

### Setup
Python version used = 3.6.5 
All library requirements are in **requirements.txt**  
1. Create an environment with the correct version of python.  
2. Install all dependencies by running the following command:
	```
	python -r requirements.txt
	```

### Feature Extraction
In **TADPOLE_D1_D2.csv** file, each row contains the data corresponding
 to a single visit of a patient.   

For each visit of a patient, we extract the following features:  
- **Image Features:** 692 columns representing MRI biomarkers, which can be 
found in the columns containing UCSFFSX (cross-sectional) and UCSFFSL (longitudinal). 
- **Cognitive Assessment Features:** ADAS11, CDRSB, MMSE, RAVLT_immediate
- **Structured Covariate Features:** AGE, PTGENDER, APOE4  

### Training
The experiment parameters can be set using the configuration file present
in the folder *configs/*. Please refer to the default config file, 
*configs/config.yaml*, for details about the model parameters. 

The **main.py** file in the *scripts/* directory contains the code that does
the following: 
1. Creates an experiment directory in the *outputs/* folder.  
2. Creates a pickle object for the data named **data.pickle** (which is stored
in the *data/* directory), that stores the features extracted from the 
**TADPOLE_D1_D2.csv** file in a structured manner, to be later used for 
training. This will be a one-time process, and the **data.pickle** file will 
be loaded for all future experiments.  
3. Trains the model.
4. Evaluates the model. 

Thus, the model can be trained by doing the following:  
	1. Set train_model = True in config.yaml.  
	2. Run the command:   
	```
	python main.py --config=../configs/config.yaml  
	```

### Experiment Outputs
The config file requires an ID to be assigned to the present experiment. This
is used to create a folder in the *outputs/* directory, which stores all the 
checkpoints, logs and results of that experiment.  

After training, the model weights are saved in the directory 
*outputs/<exp_id>/checkpoints/*.  
The loss graphs are stored in the directory 
*outputs/<exp_id>/logs/*.   
The evaluation confusion matrices are stored in the directory 
*outputs/<exp_id>/results/*.

### Evaluation
The model can be evaluated on the train, validation, and test datasets by 
doing the following:  
	1. Set train_model = False and test_model = True in config.yaml.  
	2. Run the command:  
	```
	python main.py --config=../configs/config.yaml  
	```

The results are stored in the directory *outputs/<exp_id>/results/*. 

We uploaded the experiment folder **flare_pretrained/** with the best results [here](https://www.dropbox.com/sh/vgrj13a1f0cmmcx/AADm4aHGMbLK7bCc29dsoVqma?dl=0). The model 
can be initialized with these pretrained weights by setting  
```
model.load_model: '../outputs/flare_pretrained/checkpoints/model.pth'  
```
in the config file. 

### Citation
If you find this work or dataset useful, please consider citing:
```
@article{wu2019flare,
  title={FLARe: Forecasting by Learning Anticipated Representations},
  author={Wu, Yeahuay Joie and Devarakonda, Surya Teja and Fiterau, Madalina},
  journal={arXiv preprint arXiv:1904.08930},
  year={2019}
}
```

![Hits](https://hitcounter.pythonanywhere.com/count/tag.svg?url=https://github.com/Information-Fusion-Lab-Umass/flare/tree/legacy)
