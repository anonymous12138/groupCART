# GroupCART
Implementation associated with the GroupCART paper.

## Datasets 

1. Adult Income dataset - http://archive.ics.uci.edu/ml/datasets/Adult

2. COMPAS - https://github.com/propublica/compas-analysis

3. German Credit - https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29 

4. Bank Marketing - https://archive.ics.uci.edu/ml/datasets/bank+marketing

5. Default Credit - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

6. Heart - https://archive.ics.uci.edu/ml/datasets/Heart+Disease

7. MEPS - https://meps.ahrq.gov/mepsweb/

## Data Preprocessing
* Steps for data preprocessing are suggested by [IBM AIF360](https://github.com/IBM/AIF360)
* The rows containing missing values are omitted. Non-numerical features are encoded and scaled to [0,1] range 
* For `Reweighing`, please visit [Reweighing](https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/preprocessing/reweighing.py)
