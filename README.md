<a href="https://iastro-pt.github.io/gpyrn/">
    <img align="left" width="200" height="200" src="https://i.imgur.com/8jqfpoz.png"> 
</a>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/iastro-pt/gpyrn/blob/master/LICENSE)

[![PyPI version fury.io](https://badge.fury.io/py/gpyrn.svg)](https://pypi.org/project/gpyrn/)

[![Install](https://github.com/iastro-pt/gpyrn/actions/workflows/install.yml/badge.svg)](https://github.com/iastro-pt/gpyrn/actions/workflows/install.yml)

[![Docs](https://github.com/iastro-pt/gpyrn/actions/workflows/docs.yml/badge.svg)](https://github.com/iastro-pt/gpyrn/actions/workflows/docs.yml)


<br>
<br>
<br>

# gpyrn


**Modelling stellar activity with Gaussian process regression networks**

`gpyrn` is a Python package implementing a GPRN framework for the analysis of RV
datasets.  
A GPRN is a model for multi-output regression which exploits the
structural properties of neural networks and the flexibility of Gaussian
processes.

> The GPRN was originally proposed by 
  [Wilson et al. (2012)](https://icml.cc/2012/papers/329.pdf).


### Documentation

Documentation is available [here](https://iastro-pt.github.io/gpyrn).

### Authors

The `gpyrn` package was developed at [IA](https://www.iastro.pt), in the context
of the PhD thesis of João Camacho, with contributions from João Faria and
Pedro Viana.

### Cite

If you use this package in your work, please cite the following publication
(currently under review)

```bibtex
@ARTICLE{gpyrn2022,
    author = {{Camacho}, J.~D. and {Faria}, J.~P. and {Viana}, P.~T.~P.},
        title = "{Modelling stellar activity with Gaussian process regression networks}",
    journal = {arXiv e-prints},
    keywords = {Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
        year = 2022,
        month = may,
        eid = {arXiv:2205.06627},
        pages = {arXiv:2205.06627},
archivePrefix = {arXiv},
    eprint = {2205.06627},
primaryClass = {astro-ph.EP},
    adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220506627C},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


### License

Copyright 2022 Institute of Astrophysics and Space Sciences.  
Licensed under the MIT license (see [`LICENSE`](https://github.com/iastro-pt/gpyrn/blob/main/LICENSE)).
