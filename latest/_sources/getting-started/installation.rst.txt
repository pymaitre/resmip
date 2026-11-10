

Install resmip
==============

resmip can be installed using poetry (https://python-poetry.org/).

Normal installation
-------------------

Setting up poetry
^^^^^^^^^^^^^^^^^

``poetry`` can be installed with pip:

.. code-block:: bash

    pip install poetry

Install resmip
^^^^^^^^^^^^^^

For non-development use, resmip can be installed from the project directory in two possible ways:

1. Using poetry:

.. code-block:: bash

    git clone https://github.com/pymaitre/resmip
    cd resmip
    poetry install

2. Using pip:

.. code-block:: bash

    git clone https://github.com/pymaitre/resmip
    cd resmip
    pip install .

Development
-----------

Pre-commit
^^^^^^^^^^

This library support pre-commit hook scripts (https://pre-commit.com/). ``pre-commit`` can be installed using pip:

.. code-block:: bash

    pip install pre-commit

Installation
^^^^^^^^^^^^

Install resmip with poetry. It is also recommended to install extra dependencies:

.. code-block:: bash

    git clone https://github.com/pymaitre/resmip
    cd resmip
    poetry install --with docs,test
    pre-commit install
