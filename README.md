GeoBED
======


GeoBED is a Python package for optimal experimental design tailored to geoscientific applications. It relies on the [PyTorch](https://pytorch.org) library and aims to make experiment design as easy as possible while offering a wide range of algorithms and methods to choose from.

Installation
------------


To install the package, simply run

```
    pip install git+https://github.com/dominik-strutz/GeoBED
```

> The package is still in heavy development and can change rapidly. If you want to use it, it is recommended to fix the version by running
```
    pip install git+https://github.com/dominik-strutz/GeoBED@<version>
```
Where `<version>` is the version you want to use (e.g. a commit hash or a tag).

Get started
-----------

To get started, you need to define the model prior

```python
    m_prior_dist = dist.MultivariateNormal(
        prior_mean, prior_cov
    )
```

the likelihood

```python
    def forward_function(m, design):
        ...

    def data_likelihood_func(model_samples, design):
        return dist.MultivariateNormal(forward_function(model_samples, design), sigma)
```

Those are the two main components of the model. The prior is the distribution of the model parameters before any data is observed, and the likelihood is the distribution of the data given the model parameters. The goal of the optimal experimental design is to find the design that maximizes the information gain about the model parameters given the data.

This can be done by using the `BED_base_explicit` class

```python
    BED_class = BED_base_explicit(
        m_prior_dist         = m_prior_dist,
        data_likelihood_func = data_likelihood_func,
    )
```

and by then defining possible design_points, we can calculate the information gain for each design point and choose the best one

```python
    design_points = ...

    eig, info = BED_class.calculate_EIG(
        design=design_space,
        ....
    )
```

For more information, check out the [tutorials](https://geobed.readthedocs.io/en/latest/tutorials.html) or the [API](https://geobed.readthedocs.io/en/latest/api.html) at [geobed.readthedocs.io](https://geobed.readthedocs.io/en/latest).
