# FatsMB
<div align="center">

</div>

The code is tested on NVIDIA V100 Platform.

## Quick Start

1. Download datasets refer to [MB-STR](https://github.com/yuanenming/mb-str) and [PBAT](https://github.com/TiliaceaeSU/PBAT) and put them into the `data/` folder.
2. run the model with a `yaml` configuration file like following:
```bash
python run.py fit --config src/configs/yelp.yaml
```

## Acknowledgements

Our code is based on the implementations of [MB-STR](https://github.com/yuanenming/mb-str) and [PBAT](https://github.com/TiliaceaeSU/PBAT).
