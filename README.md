# SDEGen
This is the official version of  SDEGen model! 

<div align=center>
<img src="./pic/TOC.png" width="50%" height="50%" alt="TOC" align=center />
</div>


### Training

The processed data be found here [qm9](https://doi.org/10.5281/zenodo.7938106), [drugs](https://doi.org/10.5281/zenodo.7938100) 

```python
python train.py --config_path ./bash_sde/drugs_ema.yml
python train.py --config_path ./bash_sde/qm9_default_ema.yml
```



### Generate samples for the given smiles

You can download the pre-trained checkpoint for [drugs](https://drive.google.com/file/d/1KpixpWnypOXgdF5uM7m6uNmuR8D4Nba8/view?usp=share_link) and [qm9](https://drive.google.com/file/d/14hOkQqXy_B6LxRbjk1gPt7xe3QpXQjAu/view?usp=share_link). And then put them in the corresponding directory at ./log/sde. 

To generate samples for the test set, running:

```python
python sde_sample.py --config_path bash_sde/drugs_ema.yml --num_repeat 100 --smiles C[C@H]1CCCN(C(=O)C2CCN(S(=O)(=O)c3cccc4nonc34)CC2)C1
```



### Generate samples for Platinum molecules.

Firstly, download the Platinum molecules with conformations [here](https://drive.google.com/drive/folders/15USiInCf4u8JPmRnCRcH8yb1Fzf1G_mS?usp=share_link), unzip this file to ./log/sde/platinum/opt

Then, run the following script. 

```python
python sde_sample_platinum.py --config_path bash_sde/drugs_ema.yml --start 0 --end 200 
```



### Learn more about Molecular Dynamics performed in the last part of the experiment.

I have provided the original data and the analysis.ipynb at [here](https://drive.google.com/file/d/15JuBGy1obSFm-2p8wR2iBbryXyIGMvXM/view?usp=share_link), have fun!

By the way, recently, I have noticed some people confused the molecular conformation generation and molecular generation, while the former refers to learning P(R|G), the latter is related to learning P(G).
Please be careful when conducting research! Have your own taste of work, knowing what is good and what is bad. Don't be the man that 'have no views of one's own, repeat what other says'. 
