# Heuslerene Band Exploration

This repo provides code used during the autoencoding of heuslerene band structures, and the application used for visualization of them. To access the trained model, please feel free to email me at justin.hart@ufl.edu as they are of a large size and would not fit in the repo.

# File Structure
- **BandData/**   contains band structures and data used to create them
    - **EIGENVALs/** contains VASP eigenvalues 
    - **FERMI_ENERGYs** contains fermi energy for material calculated
    - **Images/** contains all band structure plots with subfolders based on window size (&#177;x eV) and line width
    - **LargeFont_Images/** contains band structure plots with larger font sizes for easier visualization
- **Ensamble_Evaluation/** Contains code evaluating all models for the Density Based Clustering Validation score and their mean Adjusted Mutual Information score accross different HDBSCAN parameters
- **Figures/** contains images generated and used in the paper 
- **Model/** contains the ELF autoencoder from Pentz et. al. (2025)
- **Model_Evaluation/** contains files for evaluating the models after creation
    - **double_cluster.ipynb** contains code applying a second DBSCAN cluster to the UMAP proection after noise is removed.
    - **projection evaluation.ipynb** contains code evaluating HDBSCAN on a single autoencoder model
    - **unencoded_projection.ipynb** contains code showing a UMAP projection when there is no autoencoding. UMAP can find features even without autoencoding, such as in the [UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/basic_usage.html) but this is found to not be the case.
- **Model_Generation/** contains files for creating the autoencoder
    - **model_generation.ipynb** allows for training of models over different parameter spaces
- **Preprocess/** contains the code used to generate the band structures for training
    - **BandStructure2d.py** contains class for band structure
    - **CreatureBandStructures.py** actually creates the band structures.


### Credits
By Justin B. Hart, 2025.

 Note that the Autoencoder Model is from Pentz et al. at their [Elf Autoencoder](https://doi.org/10.1038/s42005-025-01936-2) paper. 
