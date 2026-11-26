# ECS708P Machine Learning 2026/27 - Coursework

## 1. Setup
`ML_CW_2026.ipynb` - includes all the implementation, experiments, outputs and analysis.
- All reqquired folders are created by the notebook. Any existing folder with the same name will be reused.

### Enviroment - Python 3.12.15 kernel and a Jupyter notebook environment.

#### Libraries - All the standard libraries are imported at the start. Some additional libraries are inclueded:
- NumPy: 2.3.5
- pandas: 2.3.3
- Matplotlib: 3.10.8
- scikit-learn: 1.8.0
- PyTorch: 2.9.1
- torchvision: 0.24.1
-  scipy, seaborn, joblib
For a new environment, run these commands from the folder containing the notebook:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==2.3.5 pandas==2.3.3 matplotlib==3.10.8 scikit-learn==1.8.0 torch==2.9.1 torchvision==0.24.1 scipy seaborn joblib
```

### Data and checkpoints
- Fashion-MNIST is downloaded automatically using `torchvision.datasets.FashionMNIST` into `data/`. Internet access is needed for the first download. Future runs will reuse the downloaded files.
- No external encoder, checkpoint or feature file are used. All MLPs are trained in the notebook.
- Part 3 creates the dropout strategy checkpoints needed by Part 5:
  `results/part3/P3-w256-e500-l0.01-dropout-seed42.pt`, with equivalent files for seeds 43 and 44.
- Part 5.1 extracts features from the seed-42 checkpoint. Part 5.2 loads all three checkpoints. Running these sections alone therefore requires the earlier outputs and the model definitions from the notebook.

## 2. Execution

Keep the notebook in the project root and start Jupyter from that folder:

```bash
jupyter notebook ML_CW_2026.ipynb
```

Select the correct Python environment, restart the kernel and run all cells from top to bottom. The notebook creates its output directories. Cells share variables and functions, so later sections are not standalone scripts.

| Order | Work performed |
| --- | --- |
| Part 0 | Download data, create fixed splits and subsets, and initialise the experiment register CSV file |
| Part 1 | Fit four classifiers, inspect errors, vary model settings to staudy and and compare decision thresholds |
| Part 2 | Train the baseline MLP and compare hidden widths and learning rates |
| Part 3 | Train nine regularisation runs: three conditions across three seeds. Save validation selected checkpoints |
| Part 4 | Search K, compare initialisation and restarts, interpret clusters and fit Ward hierarchies |
| Part 5.1 | Repeat clustering with frozen MLP features on the same images from Part 4 |
| Part 5.2 | Evaluate equal weight probability averaging on validation and then the official test set |
| Part 6 | Final summary of the evidence and limitations |

Note: A complete rerun will retrain the models and overwrite generated outputs, files and register. Keep a copy of submitted results before rerunning. One of the initial logistic regression complexity study shows a convergence warning; this is retained as a learned experiment; the notebook then repeats the fits with a larger iteration limit with code to capture any further warning.

## 3. Compute and runtimes

The development and submitted execution reports used the QMUL teaching environment: Linux x86_64, an NVIDIA A100 80 GB PCIe GPU and 96 available logical CPUs. Since its QMUL environment, the CPUs does not mean dedicated use of all cores. PyTorch models use the GPU when available; scikit-learn and scipy experiments run on the CPU.

On the teaching platform, the CUDA-enabled PyTorch environment was used. The notebook selects CUDA when available else otherwise uses the CPU. CPU execution is supported, but timings will differ.

Below are the approximate execution run timings from my latest saved notebook outputs:
| Activity | Recorded time |
| --- | --- |
| 64-unit MLP, 25 epochs, learning rate 0.005 | 12.90 seconds |
| 64-unit MLP, 25 epochs, learning rate 0.2 | 12.74 seconds |
| One MLP predicting all 10,000 test images | 0.019 to 0.035 seconds |
| Three-model ensemble predicting the same test images | 0.0845 seconds |

Part 3 regularisation experiments: Nine runs were trained for 500 epochs each: three regularisation conditions across three seeds. The records of each run's time is in `results/part3/seed_results.csv`. A complete notebook runtime was not recorded.

Warm-up and CUDA synchronisation was performed in an attempt to capture ideal performance times during inference with test set.

The ensemble needs three forward passes per image and 610,590 parameters, compared with 203,530 for one member. Its three checkpoints occupy approximately 2.34 MiB. No additional training is performed for ensembling.

## 4. Reproducibility

### Data splits and indices

| Stored record | Purpose |
| --- | --- |
| `configs/fashion_mnist_split_config.json` | Stratified 50,000/10,000 training and validation split with seed 42; fixed 10,000 image fitting subset from training with seed 43 |
| `configs/fashion_mnist_cluster_config.json` | Fixed clustering subsets: 3,000 fitting images and 1,000 validation images witg seed 44 |
| `results/part4_1/used_indices.json` | Exact training and validation indices reused for the raw-pixel and MLP feature comparisons |
| `results/part4_4/image_indices.npy` | The 200 training images sampled without replacement for both hierarchical comparisons with seed 42 |

Training and validation indices refer to the official Fashion-MNIST training set. Test indices refer to its separate test set
The Part 5.2 evaluates all the 10,000 from official test images. It is not using the clustering test subset.


### Experiment records and outputs

`results/experiment_register.csv` file contain the records of experiments per coursework including questions, hypothesis, settings, seeds and output locations etc. 
Figures and tables are displayed in the notebook, with saved outputs under `figures/` and `results/`. 
`figures/` only includes figures from Part 1; Figures for remains parts are included within their respective sub-folders under `results/`.

## 5. Acknowledgements

- No externally pre-trained checkpoint was used. The extension reuses the three MLPs trained in Part 3.
- OpenAI ChatGPT GPT-6 Astra assistance was used for the following:
 - learning and explanations: why and how-to use cross-entropy validation loss for checkpoints rather than just accuracy; concepts and   approach to kmeans++ grouping and how to to use ward linkage for hierarchical clustering; some related graph interpretations
 - experiment planning: how and why to use both weight decay and dropout strategy; planning to use Part 3 checkpoints in Part 5
 - code development and debugging: code generation for sub-plots; smartly displaying images and captions; debugging convergence warning
 - written analysis: calculating % differences across parts; shortening my long sentences in few cases

