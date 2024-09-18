FORCE
=====

Forecasting Offshore wind Reductions in Cost of Energy

:Version: 0.1
:Authors: `Jake Nunemaker <https://www.linkedin.com/in/jake-nunemaker/>`_, `Matt Shields <https://www.linkedin.com/in/matt-shields-834a6b66/>`_, `Philipp Beiter <https://www.linkedin.com/in/philipp-beiter-365b4189/>`_

Installation
------------

1. Clone the repository.
2. Create a new conda environment. ``conda create --name force python=3.9``
3. Activate the new environment.
4. Install FORCE with the setup script. ``pip install -e .``

Dependencies
~~~~~~~~~~~~

- numpy
- pandas
- scipy
- matplotlib
- pandas
- pyyaml
- xlsxwriter
- orbit-nrel

Recommended Packages
~~~~~~~~~~~~~~~~~~~~

- Jupyter Lab

Project Database
----------------

This project was designed using the 4C Project Database. As this is a paid
subscription, this dataset is not included in this repo. If you are a 4C
customer and have access to the project dataset, it can be placed in the
``analysis/data/`` folder and be used for the regression analysis. Please don't
commit this file to the repo.

If you do not have access to this dataset, a template file is provided that can
be filled out with other project data and used with this package. This template
is located at ``analysis/data/project_list_template.csv``. Note: Not all columns
in the template are required but FORCE can be configured to use them in the
regression analysis.
