# Heuslerene
A curated database of Heusler alloys and their predicted 2D equivalents, "Heuslerenes." Includes structural data, stability analysis, and deep learning code for accelerating research on and exploring 2D Heusler materials. 

# File Structure
- **Autoencoder/** contains the code used to train the Machine Learning model described in the supplementary materials. Additionally, a README.md file in this folder explains its file structure.
- **Crystal Data/**
    - **cystal-structures/** contains the VASP CONTCAR files for the Heuslerenes studied
    - **EIGENVALs/** contains the VASP EIGENVAL files for the Hueslerenes studied
    - **FERMI_ENERGYs/** contains the Fermi energy values for the Heuslerenes studied. Necessary to create an electronic band shifted to the Fermi energy.
    - **Figures-PBE-SOC-BandStructure** contains images of the band structures. These can also be explored in the link below.
- **explorer/** contains a web application for exploring the band structures. A README.md file in this folder explains how to run it on your local system.


## 📊 Visualizations

[View PBE+SOC Bandstructures](https://sriharikastuar.github.io/Heuslerene/Figures-PBE-SOC-BandStructure/gallery.html)

## Credits

By Srihari M. Kastuar, Justin B. Hart, Anthony C. Iloanya, and Chinedu E. Ekuma, 2025.
