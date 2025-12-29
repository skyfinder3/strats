## AmsterdamUMCdb implementation

set up local environment. 
Install the requirements in `requirements.txt`
for convenience we have 2 different setups with at gtx 1080 and 5070, which need different pytorch packages.


## get import dataset

import the dataset to data\custom\Matias\df_long.csv
(sorry not cleaned up yet)

**Execute STraTS**
open a shell and Execure run_aumc.sh


Execute I-Strats**
open a shell end execute run_aumc_istrats.sh

This will execute the whole pipeline including dataset creation, training and evaluation

the models and all outputs will be found in outputs/[dataset]

## extra
For istrats run change the path to the output.pt in `custom_analyze_istrats.py` to the path of the .pt output to see the variable contributions