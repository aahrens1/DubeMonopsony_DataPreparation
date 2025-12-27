### Replication of Dube et al. 

[This commit](https://github.com/aahrens1/DubeMonopsony_DataPreparation/commit/882cd2c903dea8cd5a7b22cb8d52fdc7413a9596) documents changes compared to original replication code. These were required to make the code run on newer Python versions.


1. Download replication code from [AEA website](https://www.aeaweb.org/articles?id=10.1257/aeri.20180150)
2. Replace folder `double_ml_code` with code in this repository
3. Use `environment.yml` to set up Python environment and active the environment

For example, using mamba/conda:

```
mamba env create -f environment.yml
conda activate py39text
```

4. Generate replication files using

```
python ml_pipeline.py ipeirotis 
```