# SUMO - Slim U-Net trained on MODA

Implementation of the *SUMO* (*S*lim *U*-Net trained on *MO*DA) model as described in:

```
Lars Kaulen, Justus T.C. Schwabedal, Jules Schneider, Philipp Ritter and Stephan Bialonski.
Advanced sleep spindle identification with neural networks. Sci Rep 12, 7686 (2022).
https://doi.org/10.1038/s41598-022-11210-y
```

## Installation Guide

On Linux and Windows the project can be used by running the following commands to clone the repository and install the required dependencies.

### Either with `anaconda` or `miniconda` installed
```shell
git clone https://github.com/dslaborg/sumo.git
cd sumo
conda env create --file environment.yaml
conda activate sumo
```
### or using `pip`
```shell
git clone https://github.com/dslaborg/sumo.git
cd sumo
pip install -r requirements.txt
```

### `pip` with virtualenv for **Linux/MacOS**
```shell
# enter the project root
git clone https://github.com/dslaborg/sumo.git
cd sumo

# Create the virtual env with pip
virtualenv venv --python=python3.9.10
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# With jupyter notebook
pip install ipykernel
pip install ipywidgets
```

### `pip` with virtualenv for **Windows**
```shell
# enter the project root
git clone https://github.com/dslaborg/sumo.git
cd sumo

# Create the virtual env with pip
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# With jupyter notebook
pip install ipykernel
pip install ipywidgets
```

## Using the SUMO model - TL;DR

When only wanting to use the already trained SUMO model (see `output/final.ckpt`) to detect spindles on unknown data, the first entry point is the `scripts/predict_plain_data.py` file.

There, any EEG data - given as a dict with the channel name as key and the EEG channel as value in .pickle or .npy format - can be used to predict spindles on.
The necessary preprocessing steps (downsampling, passband Butterworth filtering and z-transformation) are included, as well as a showcase on how to use the applied Pytorch Lightning framework to predict spindles on the transformed data.

Next to the `sumo` package, containing the model and necessary functions and classes, only the configuration files `config/default.yaml` and `config/predict.yaml`, the model checkpoint `output/final.ckpt` and the input data (e.g. `input/eeg-sample.npy`) are needed to run the script.

## Scripts

### Running and evaluating an experiment

The main model training and evaluation procedure is implemented in `bin/train.py` and `bin/eval.py` using the [Pytorch Lightning](https://www.pytorchlightning.ai/) framework.
A chosen configuration used to train the model is called an *experiment*, and the evaluation is carried out using a configuration and the result folder of a training run.

#### train.py

Trains the model as specified in the corresponding configuration file, writes its log to the console and saves a log file and intermediate results for Tensorboard and model checkpoints to a result directory.

Arguments:
* `-e NAME, --experiment NAME`: name of experiment to run, for which a `NAME.yaml` file has to exist in the `config` directory; default is `default`

#### eval.py

Evaluates a trained model, either on the validation data or test data and reports the achieved metrics.

Arguments:
* `-e NAME, --experiment NAME`: name of configuration file, that should be used for evaluation, for which a `NAME.yaml` file has to exist in the `config` directory; usually equals the experiment used to train the model; default is `default`
* `-i PATH, --input PATH`: path containing the model that should be evaluated; the given input can either be a model checkpoint, which then will be used directly, or the output directory of a `train.py` execution, in which case the best model will be used from `PATH/models/`; if the configuration has cross validation enabled, the output directory is expected and the best model per fold will be obtained from `PATH/fold_*/models/`; no default value
* `-t, --test`: if given, the test data is used instead of the validation data

### Further example scripts

In addition to scripts used to create the figures in our manuscript (`spindle_analysis.py`, `spindle_analysis_correlations.py` and `spindle_detection_examply.py`), the `scripts` directory contains two scripts that demonstrate the usage of this project.

#### create_data_splits.py

Demonstrates the procedure used to split the data into test and non-test subjects and the subsequent creation of a hold-out validation set and (*alternatively*) cross validation folds.

Arguments:
* `-i PATH, --input PATH`: path containing the (necessary) input data, as produced by the MODA file [MODA02_genEEGVectBlock.m](https://github.com/klacourse/MODA_GC/blob/master/MODA02_genEEGVectBlock.m); default is `<project-dir>/input/`
* `-o PATH, --output PATH`: path in which the generated data splits should be stored in; default is `<project-dir>/output/datasets_{datatime}`
* `-n NUMBER, --n_datasets NUMBER`: number of random split-candidates drawn/generated; default is `25`
* `-t FRACTION, --test FRACTION`: Proportion of data that is used as test data; `0<=FRACTION<=1`; default is `0.2`

#### predict_plain_data.py

Demonstrates how to predict spindles with a trained SUMO model on arbitrary EEG data, which is expected as a dict with the keys representing the EEG channels and the values the corresponding data vector.

Arguments:
* `-d PATH, --data_path PATH`: path containing the input data, either in `.pickle` or `.npy` format, as a dict with the channel name as key and the EEG data as value; default is `<project-dir>/input/eeg-sample.npy` (synthetic data)
* `-m PATH, --model_path PATH`: path containing the model checkpoint, which should be used to predict spindles; default is `<project-dir>/output/final.ckpt`
* `-g NUMBER, --gpus NUMBER`: number of GPUs to use, if `0` is given, calculations are done using CPUs; default is `0`
* `-sr RATE, --sample_rate RATE`: sample rate of the provided data; default is `100.0`

## Project Setup

The project is set up as follows:

* `bin/`: contains the `train.py` and `eval.py` scripts, which are used for model training and subsequent evaluation in experiments (as configured within the `config` directory) using the [Pytorch Lightning](https://www.pytorchlightning.ai/) framework
* `config/`: contains the configurations of the experiments, configuring how to train or evaluate the model
  * `default.yaml`: provides a sensible default configuration
  * `final.yaml`: contains the configuration used to train the final model checkpoint (`output/final.ckpt`)
  * `predict.yaml`: configuration that can be used to predict spindles on arbitrary data, e.g. by using the script at `scripts/predict_plain_data.py`
* `input/`: should contain the used input files, e.g. the EEG data and annotated spindles as produced by the [MODA repository](https://github.com/klacourse/MODA_GC/blob/master/MODA02_genEEGVectBlock.m) and transformed as demonstrated in the `/scripts/create_data_splits.py` file
* `output/`: contains generated output by any experiment runs or scripts, e.g. the created figures
  * `final.ckpt`: the final model checkpoint, on which the test data performance, as reported in the paper, was obtained
* `scripts/`: various scripts used to create the plots of our paper and to demonstrate the usage of this project
  * `a7/`: python implementation of the A7 algorithm as described in:
    ```
    Karine Lacourse, Jacques Delfrate, Julien Beaudry, Paul E. Peppard and Simon C. Warby. "A sleep spindle detection algorithm that emulates human expert spindle scoring." Journal of Neuroscience Methods 316 (2019): 3-11.
    ```
  * `create_data_splits.py`: demonstrates the procedure, how the data set splits were obtained, including the evaluation on the A7 algorithm
  * `predict_plain_data.py`: demonstrates the prediction of spindles on arbitrary EEG data, using a trained model checkpoint
  * `spindle_analysis.py`, `spindle_analysis_correlations.py`, `spindle_detection_example.py`: scripts used to create some of the figures used in our paper
* `sumo/`: the implementation of the SUMO model and used classes and functions, for more information see the docstrings

## Configuration Parameters

The configuration of an experiment is implemented using yaml configuration files.
These files must be placed within the `config` directory and must match the name past as `--experiment` to the `eval.py` or `train.py` script.
The `default.yaml` is always loaded as a set of default configuration parameters and parameters specified in an additional file overwrite the default values.
Any parameters or groups of parameters that should be `None`, have to be configured as either `null` or `Null` following the YAML definition.

The available parameters are as follows:
* `data`: configuration of the used input data; optional, can be `None` if spindle should be annotated on arbitrary EEG data
  * `directory` and `file_name`: the input file containing the `Subject` objects (see `scripts/create_data_splits.py`) is expected to be located at `${directory}/${file_name}`; the file should be a (pickled) dict with the name of a data set as key and the list of corresponding subjects as value; default is `input/subjects.pickle`
  * `split`: describing the keys of the data sets to be used, specifying either `train` and `validation`, or `cross_validation`, and optionally `test`
    * `cross_validation`: can be either an integer k>=2, in which the keys `fold_0`, ..., `fold_{k-1}` are expected to exist, or a list of keys
  * `batch_size`: size of the used minbatches during training; default is `12`
  * `preprocessing`: if z-scoring should be performed on the EEG data, default is `True`
* `experiment`: definition of the performed experiment; mandatory
  * `model`: definition of the model configuration; mandatory
    * `n_classes`: number of output parameters; default is `2`
    * `activation`: name of an activation function as defined in `torch.nn` package; default is `ReLU`
    * `depth`: number of layers of the U excluding the *last* layer; default is `2`
    * `channel_size`: number of filters of the convolutions in the *first* layer; default is `16`
    * `pools`: list containing the size of pooling and upsampling operations; has to contain as many values as the value of `depth`; default `[4;4]`
    * `convolution_params`: parameters used by the Conv1d modules
    * `moving_avg_size`: width of the moving average filter; default is `42`
  * `train`: configuration used in training the model; mandatory
    * `n_epochs`: maximal number of epochs to be run before stopping training; default is `800`
    * `early_stopping`: number of epochs without any improvement in the `val_f1_mean` metric, after which training is stopped; default is `300`
    * `optimizer`: configuration of an optimizer as defined in `torch.optim` package; contains `class_name` (default is `Adam`) and parameters, which are passed to the constructor of the used optimizer class
    * `lr_scheduler`: used learning rate scheduler; optional, default is `None`
    * `loss`: configuration of loss function as defined either in `sumo.loss` package (`GeneralizedDiceLoss`) or `torch.nn` package; contains `class_name` (default is `GeneralizedDiceLoss`) and parameters, which are passed to the constructor of the used loss class
  * `validation`: configuration used in evaluating the model; mandatory
    * `overlap_threshold_step`: step size of the overlap thresholds used to calculate (validation) F1 scores
