import matplotlib.pyplot as plt
import os
import numpy as np
import re

class BandStructure2d():
    """ Contains files to parse Eigenval and Fermi Energies, creating a band structure in a desired
    range to help in autoencoding
    """

    def __init__(self, 
                 element, 
                 eigenval_path=r'../BandData/EIGENVALs', 
                 fermi_energy_path=r'../BandData/FERMI_ENERGYs'
                 ):
        self.element = element
        self.element_special = re.sub(r'(\d+)', r'$_\1$', element)
        self.eigenval_path = eigenval_path
        self.fermi_energy_path = fermi_energy_path

        self.fermi_energy = self._get_fermi_energy()
        self.kpoints, self.eigenvalues = self._parse_eigenvals()
        self.distances = self._compute_distances()


    def _get_fermi_energy(self):
        """returns Fermi energy retrived from provided file"""
        path = os.path.join(self.fermi_energy_path, f"2D_{self.element}_EF")
        with open(path, 'r') as f:
            lines = f.readlines()
            fermi_energy = float(lines[1].split()[0])    
        
        return fermi_energy


    def _parse_eigenvals(self):
        """ Parses given eigenval file for the current elememnt, returning an array of kpoints and an
        array of eigenvalues
        """
        path = os.path.join(self.eigenval_path, f"2D_{self.element}_EIGENVAL")
        
        with open(path, 'r') as f:
            lines = f.readlines()

        nkpts = int(lines[5].split()[1])
        nbands = int(lines[5].split()[2])

        kpoints = []
        eigenvals = []

        for i in range(7,len(lines), nbands + 2):
            kpt = list(map(float, lines[i].split()[:3]))
            kpoints.append(kpt)
            
            bands = []
            for j in range(nbands):
                bands.append(float(lines[i + j + 1].split()[1]))

            eigenvals.append(bands)

        return np.array(kpoints), np.array(eigenvals)
    


    def _compute_distances(self):
        distances = [0.0]
        for i in range(1, len(self.kpoints)):
            dk = np.linalg.norm(self.kpoints[i]-self.kpoints[i-1])
            distances.append(distances[-1]+dk)

        return np.array(distances)


    def plot(self, energy_window=(-3,3), sym_points=None, color='black', linewidth=0.5, save_path=None):
        """ return a MatPlotLib figure of the band structure in a set range of the fermi energy
        """

        eig = self.eigenvalues - self.fermi_energy
        fig, ax = plt.subplots()

        for band in eig.T:
            ax.plot(self.distances, band, color=color, linewidth=linewidth)

        ax.set_ylim(*energy_window)
        ax.axhline(0, color='red', linestyle='--', linewidth=0.5)

        ax.set_xlim(self.distances[0], self.distances[-1])


        if sym_points:
            # indices that mark Γ, M, K, Γ  (0‑based)
            idx_sym = [0, 100, 200, len(self.distances)-1]
            x_sym   = [self.distances[i] for i in idx_sym]    # true x‑coordinates
            labels  = [r"$\Gamma$", "M", "K", r"$\Gamma$"]

            ax.set_xticks(x_sym)
            ax.set_xticklabels(labels)
            for x in x_sym[1:-1]:
                ax.axvline(x, color="0.7", linewidth=0.4, zorder=0)


        ax.set_ylabel("Energy (eV)")
        ax.set_xlabel("k-path")
        ax.set_title(f"Band Structure: {self.element}")

        if save_path:
            fig.savefig(os.path.join(save_path, f"{self.element}.png"), pad_inches=0)
       
        plt.close(fig)

        return fig
    
    def plot_large(self, energy_window=(-3,3), sym_points=None, color='black', 
                   linewidth=1, save_path=None, title_size=12, text_size=10, number_size=10):
        """ return a MatPlotLib figure of the band structure in a set range of the fermi energy
        """

        eig = self.eigenvalues - self.fermi_energy
        fig, ax = plt.subplots()

        for band in eig.T:
            ax.plot(self.distances, band, color=color, linewidth=linewidth)

        ax.set_ylim(*energy_window)
        ax.tick_params(axis='y', labelsize=number_size, width=2)
        ax.axhline(0, color='red', linestyle='--', linewidth=linewidth)

        ax.set_xlim(self.distances[0], self.distances[-1])
        ax.spines['bottom'].set_linewidth(2)  # Set x-axis linewidth to 2
        ax.spines['left'].set_linewidth(2) 
        ax.spines['top'].set_linewidth(2) 
        ax.spines['right'].set_linewidth(2) 

        if sym_points:
            # indices that mark Γ, M, K, Γ  (0‑based)
            idx_sym = [0, 100, 200, len(self.distances)-1]
            x_sym   = [self.distances[i] for i in idx_sym]    # true x‑coordinates
            labels  = [r"$\Gamma$", "M", "K", r"$\Gamma$"]

            ax.set_xticks(x_sym)
            ax.set_xticklabels(labels, fontsize=text_size)
            for x in x_sym[1:-1]:
                ax.axvline(x, color="0.7", linewidth=linewidth*1.5, zorder=0)


        ax.set_ylabel("Energy (eV)",fontsize=text_size)
        ax.set_title(f"{self.element_special}", fontsize=title_size)

        if save_path:
            fig.savefig(os.path.join(save_path, f"{self.element}.png"), pad_inches=0)
       
        plt.close(fig)

        return fig
    

    def plot_for_ml(self, energy_window=(-3,3), figsize_pixel=(224,224), save_path=None, linewidth=0.5):
        eig = self.eigenvalues - self.fermi_energy

        width_px, height_px = figsize_pixel
        dpi = 100

        figsize = (width_px / dpi, height_px / dpi)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_position([0, 0, 1, 1]) 


        for band in eig.T:
            ax.plot(self.distances, band, color='black', linewidth=linewidth, antialiased=False)

        ax.set_ylim(*energy_window)
        
        #ax.axhline(0, color='black', linestyle='--', linewidth=0.5) #places line to show fermi energy

        ax.set_xlim(self.distances[0], self.distances[-1])

        ax.axis('off')
        
        if save_path:
            fig.savefig(os.path.join(save_path, f"{self.element}.png"), dpi=dpi, pad_inches=0)
       
        plt.close(fig)

        return fig
