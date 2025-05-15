First, download the dataset. For that you will need `huggingface-cli`

```
dataset.sh
```

To set up the environment use conda

```
conda env create -f environment.yml
conda activate transformers
```

to prepare the tokens run

```
python tokenizer.py -p data.txt -d dataset.csv --vocab 20000 --d_model 512 --min_occurrence 1e-6
```

then run the `train` script by

```
python train_translator.py
```
