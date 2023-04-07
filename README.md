# A Clip-Hitchhiker's Guide to Long Video Retrieval

Implementation of query-scoring in the [Clip-Hitchhiker](https://arxiv.org/abs/2205.08508) technical report, Bain et al., Arxiv 2022.


*N.B.: This paper was unfortunately rejected from ECCV 2022, and we decided not to re-submit and instead work on other things. Nonetheless, I release the code and ArXiv paper, in case you might find some of the ideas and code in the work useful.*


### Install

First, install PyTorch 1.7.1 (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install pandas numpy decord
$ pip install git+https://github.com/openai/CLIP.git
```
### Method
query_scoring.py contains `def similarity_queryscoring` function which is the proposed method in the paper.


### Zero-shot eval
Run query-scoring method with CLIP on MSR-VTT text-to-video retrieval.
N.B.: Downloads 6GB MSR-VTT videos to ./data

`python run_msr_retrieval.py --num_frames 16`

you can play around with different temperature values (tau), when temperature << 1, it approximates argmax, when temperature >> 1 it approximates uniform mean-pooling.

for best results, use 120 frames (takes longer, and needs more RAM, chunking required)
`python run_msr_retrieval --num_frames 120 --batch_size 1 --num_workers 1`

expected results 16 frames:

```
{'R1': 33.3, 'R5': 55.0, 'R10': 65.2, 'R50': 84.8, 'MedR': 4.0, 'MeanR': 38.403}  agg: mean-pooling
{'R1': 34.4, 'R5': 56.1, 'R10': 65.3, 'R50': 85.5, 'MedR': 4.0, 'MeanR': 37.602}  agg: query-scoring, temp: 0.1
```

### Finetuning
We do not provide finetuning code, however the query_scoring function can be plugged into the loss function when calculating Video-Text Contrastive loss.


### Citation

```bibtex
@article{bain2022clip,
  title={A CLIP-Hitchhiker's Guide to Long Video Retrieval},
  author={Bain, Max and Nagrani, Arsha and Varol, G{\"u}l and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2205.08508},
  year={2022}
}
```
