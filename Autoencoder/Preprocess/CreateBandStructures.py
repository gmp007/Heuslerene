from BandStructure2d import BandStructure2d
import os

path = r"../BandData"

width = 1
window = 1
width_px = 224

output_path = os.path.join(path, 'Images', f'bands_{int(width * 10)}width_{window}eV',)
output_path_large = os.path.join(path, 'LargeFont_Images', f'bands_{int(2*width * 10)}width_{window}eV',)
output_path_ml = os.path.join(path, 'Images', f'bands_{int(width * 10)}width_{window}eV_ml', 'class')

os.makedirs(output_path, exist_ok=True)
os.makedirs(output_path_large,  exist_ok=True)
os.makedirs(output_path_ml,  exist_ok=True)

for filename in os.listdir(os.path.join(path, 'EIGENVALs')):
    file_path = os.path.join(path, 'EIGENVALs', filename)
    if os.path.isfile(file_path):
        chemical = file_path.split(os.sep)[-1].split('_')[1]
        try:
            band = BandStructure2d(chemical)
            """
            # Plotting the Training Data
            band.plot_for_ml(energy_window=(-window,window), 
                             figsize_pixel=(width_px, width_px), 
                             save_path=output_path_ml,
                             linewidth=width)

            # Plotting Visualization
            band.plot(energy_window=(-window,window), 
                             sym_points=True,
                             save_path=output_path,
                             linewidth=width,
                             )
            """

            # Plotting Visualization with Larger Fonts
            band.plot_large(energy_window=(-window,window), 
                             sym_points=True,
                             save_path=output_path_large,
                             linewidth=2*width,
                             title_size=26,
                             text_size=14,
                             number_size=10
                             )
                             
        except FileNotFoundError as f:
            print(f)






