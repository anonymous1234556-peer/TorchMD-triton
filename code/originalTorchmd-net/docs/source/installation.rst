Installation
============

TorchMD-Net is available as a pip installable wheel as well as in `conda-forge <https://conda-forge.org/>`_

TorchMD-Net provides builds for CPU-only, CUDA 11.8 and CUDA 12.4. 
CPU versions are only provided as reference, as the performance will be extremely limited.
Depending on which variant you wish to install, you can install it with one of the following commands:

.. code-block:: shell

   # The following will install the CUDA 12.4 version by default
   pip install torchmd-net 
   # The following will install the CUDA 11.8 version
   pip install torchmd-net --extra-index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://us-central1-python.pkg.dev/pypi-packages-455608/cu118/simple
   # The following will install the CUDA 12.4 version
   pip install torchmd-net --extra-index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://us-central1-python.pkg.dev/pypi-packages-455608/cu124/simple
   # The following will install the CPU only version (not recommended)
   pip install torchmd-net --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://us-central1-python.pkg.dev/pypi-packages-455608/cpu/simple   


Alternatively it can be installed with conda or mamba with one of the following commands.
We recommend using `Miniforge <https://github.com/conda-forge/miniforge/>`_ instead of anaconda.

.. code-block:: shell

   mamba install torchmd-net cuda-version=11.8
   mamba install torchmd-net cuda-version=12.4


Install from source
-------------------

1. Clone the repository:

.. code-block:: shell

   git clone https://github.com/torchmd/torchmd-net.git
   cd torchmd-net

2. Install the dependencies in environment.yml.

.. code-block:: shell

   conda env create -f environment.yml
   conda activate torchmd-net

3. CUDA enabled installation

You can skip this section if you only need a CPU installation.

You will need the CUDA compiler (nvcc) and the corresponding development libraries to build TorchMD-Net with CUDA support. You can install CUDA from the `official NVIDIA channel <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation>`_ or from conda-forge.

The conda-forge channel `changed the way to install CUDA from versions 12 and above <https://github.com/conda-forge/conda-forge.github.io/issues/1963>`_, thus the following instructions depend on whether you need CUDA < 12. If you have a GPU available, conda-forge probably installed the CUDA runtime (not the developer tools) on your system already, you can check with conda:
   
.. code-block:: shell

   conda list | grep cuda

   
Or by asking pytorch:
   
.. code-block:: shell
		 
   python -c "import torch; print(torch.version.cuda)"

   
It is recommended to install the same version as the one used by torch.  

.. warning:: At the time of writing there is a `bug in Mamba <https://github.com/mamba-org/mamba/issues/3120>`_ (v1.5.6) that can cause trouble when installing CUDA on an already created environment. We thus recommend conda for this step.
	     
* CUDA>=12

.. code-block:: shell

   conda install -c conda-forge python=3.10 cuda-version=12.6 cuda-nvvm cuda-nvcc cuda-libraries-dev


* CUDA<12  
  
The nvidia channel provides the developer tools for CUDA<12.
  
.. code-block:: shell
		 
   conda install -c nvidia "cuda-nvcc<12" "cuda-libraries-dev<12" "cuda-version<12" "gxx<12" pytorch=*=*cuda*


4. Install TorchMD-NET into the environment:

.. code-block:: shell

   pip install -e .


.. note:: Pip installation in CUDA mode requires compiling CUDA source codes, this can take a really long time and the process might appear as if it is "stuck". Run pip with `-vv` to see the compilation process.

This will install TorchMD-NET in editable mode, so that changes to the source code are immediately available.
Besides making all python utilities available environment-wide, this will also install the ``torchmd-train`` command line utility.

