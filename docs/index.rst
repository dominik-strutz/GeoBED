GeoBED
======


GeoBED is a Python package for optimal experimental design taylored to geoscientific applications. It relies on the `PyTorch <https://pytorch.org>`_ library and aims to make the design of experiments as easy as possible, while having a wide raange of algorithms and methods to choose from.

Installation
------------


To install the package, simply run

.. code-block:: console

    pip install git+https://github.com/dominik-strutz/GeoBED


.. note::
    The package is still in heavy development and can change rapidly. If you want to use it, its recommended to fix the version by running

    .. code-block:: console

        pip install git+https://github.com/dominik-strutz/GeoBED@<version>

    where `<version>` is the version you want to use (e.g. a commit hash or a tag).

Get started
-----------

To get started you needd to define the model prior

.. code-block:: python

    m_prior_dist = dist.MultivariateNormal(
        prior_mean, prior_cov
    )
    
the likelihood

.. code-block:: python

    def forward_function(m, design):
        ...

    def data_likelihood_func(model_samples, design):
    
    return dist.MultivariateNormal(forward_function(model_samples, design), sigma)

Those are the two main components of the model. The prior is the distribution of the model parameters before any data is observed. The likelihood is the distribution of the data given the model parameters. The goal of the optimal experimental design is to find the design that maximizes the information gain about the model parameters given the data.

This can be done by using the `BED_base_explicit` class

.. code-block:: python

    BED_class = BED_base_explicit(
        m_prior_dist         = m_prior_dist,
        data_likelihood_func = data_likelihood_func,
    )

and by then defining possible design_points we can calculate the information gain for each design point and choose the best one

.. code-block:: python

    design_points = ...

    eig, info = BED_class.calculate_EIG(
        design=design_space,
        ....
    )

For more information, check out the :doc:`tutorials <../tutorials>` or the :doc:`API <../api>`.


.. toctree::
    :caption: geobed
    :hidden:
    :maxdepth: 3

    tutorials.rst
    api.rst

.. toctree::
    :caption: Development
    :hidden:
    :maxdepth: 2

    License <https://github.com/dominik-strutz/GeoBED/blob/main/LICENSE>