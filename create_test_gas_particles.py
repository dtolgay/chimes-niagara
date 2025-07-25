import sys
import h5py
import numpy as np
import pandas as pd 

sys.path.append('/scratch/m/murray/dtolgay')
from tools import constants # type: ignore


def main():

    solar_mass_fraction_abundances = {
        "H": 0.709,
        "He": 0.281,
        "C": 0.002,
        "N": 0.001,
        "O": 0.005,
        "Ne": 0.001,
        "Mg": 0.0007,
        "Si": 0.0004,
        "S": 0.0007,
        "Ca": 0.0002,
        "Fe": 0.0017
    }

    metallicity = constants.solar_metallicity # solar metallicity -- mass ratio
    

    gas_properties = {
        "hden": 100, # cm-3 
        "temperature": 300, # K
        "metallicity": metallicity, # solar metallicity -- mass ratio
        "He_mass_fraction": solar_mass_fraction_abundances["He"] * metallicity,
        "C_mass_fraction": solar_mass_fraction_abundances["C"] * metallicity,
        "N_mass_fraction": solar_mass_fraction_abundances["N"] * metallicity,
        "O_mass_fraction": solar_mass_fraction_abundances["O"] * metallicity,
        "Ne_mass_fraction": solar_mass_fraction_abundances["Ne"] * metallicity,
        "Mg_mass_fraction": solar_mass_fraction_abundances["Mg"] * metallicity,
        "Si_mass_fraction": solar_mass_fraction_abundances["Si"] * metallicity,
        "S_mass_fraction": solar_mass_fraction_abundances["S"] * metallicity,
        "Ca_mass_fraction": solar_mass_fraction_abundances["Ca"] * metallicity,
        "Fe_mass_fraction": solar_mass_fraction_abundances["Fe"] * metallicity,
    }

    log10_Nh_range = np.linspace(14, 24, 1000) # cm-3

    constructed_particles = {
        "Nh": 10**log10_Nh_range, # cm-2
        "hden": gas_properties["hden"] * np.ones_like(log10_Nh_range), # cm-3
        "temperature": gas_properties["temperature"] * np.ones_like(log10_Nh_range), # K
        "metallicity": gas_properties["metallicity"] * np.ones_like(log10_Nh_range), # solar metallicity -- mass ratio
        "He_mass_fraction": gas_properties["He_mass_fraction"] * np.ones_like(log10_Nh_range),
        "C_mass_fraction": gas_properties["C_mass_fraction"] * np.ones_like(log10_Nh_range),
        "N_mass_fraction": gas_properties["N_mass_fraction"] * np.ones_like(log10_Nh_range),
        "O_mass_fraction": gas_properties["O_mass_fraction"] * np.ones_like(log10_Nh_range),
        "Ne_mass_fraction": gas_properties["Ne_mass_fraction"] * np.ones_like(log10_Nh_range),
        "Mg_mass_fraction": gas_properties["Mg_mass_fraction"] * np.ones_like(log10_Nh_range),
        "Si_mass_fraction": gas_properties["Si_mass_fraction"] * np.ones_like(log10_Nh_range),
        "S_mass_fraction": gas_properties["S_mass_fraction"] * np.ones_like(log10_Nh_range),
        "Ca_mass_fraction": gas_properties["Ca_mass_fraction"] * np.ones_like(log10_Nh_range),
        "Fe_mass_fraction": gas_properties["Fe_mass_fraction"] * np.ones_like(log10_Nh_range),
    }

    # Convert the dictionary to a dataframe 
    constructed_particles = pd.DataFrame(constructed_particles)

    write_file_path = "/scratch/m/murray/dtolgay/post_processing_fire_outputs/chimes/hdf5_files/test/Zsolar_nh100_T300_varyingNh.hdf5"
    write_to_hdf5_file(constructed_particles, write_file_path)

    return None


def write_to_hdf5_file(constructed_particles, write_file_path):

    metal_fractions_array = np.array([
        constructed_particles['metallicity'].to_numpy(dtype=np.float64),
        constructed_particles['He_mass_fraction'].to_numpy(dtype=np.float64),
        constructed_particles['C_mass_fraction'].to_numpy(dtype=np.float64),
        constructed_particles['N_mass_fraction'].to_numpy(dtype=np.float64),
        constructed_particles['O_mass_fraction'].to_numpy(dtype=np.float64),
        constructed_particles['Ne_mass_fraction'].to_numpy(dtype=np.float64),
        constructed_particles['Mg_mass_fraction'].to_numpy(dtype=np.float64),
        constructed_particles['Si_mass_fraction'].to_numpy(dtype=np.float64),
        constructed_particles['S_mass_fraction'].to_numpy(dtype=np.float64),
        constructed_particles['Ca_mass_fraction'].to_numpy(dtype=np.float64),
        constructed_particles['Fe_mass_fraction'].to_numpy(dtype=np.float64)
    ]).T # Transpose to get the correct shape

    with h5py.File(write_file_path, 'w') as f:
        
        ##### Gas Particles ##### 
        
        # Create group for gas particles
        part_group_0 = f.create_group('PartType0')

        # Store the 3D vectors (shape: [N, 3])
        # part_group_0.create_dataset('p', data=constructed_particles[['x', 'y', 'z']].to_numpy(dtype=np.float64))                       # pc 
        # part_group_0.create_dataset('v', data=constructed_particles[['vx', 'vy', 'vz']].to_numpy(dtype=np.float64))                    # km/s

        # Scalar fields
        # part_group_0.create_dataset('m', data=constructed_particles['mass'].to_numpy(dtype=np.float64))                                # Msolar
        # part_group_0.create_dataset('rho', data=constructed_particles['density'].to_numpy(dtype=np.float64))                           # gr/cm^3
        # part_group_0.create_dataset('h', data=constructed_particles['smoothing_length'].to_numpy(dtype=np.float64))                    # pc
        # part_group_0.create_dataset('sfr', data=constructed_particles['star_formation_rate'].to_numpy(dtype=np.float64))               # sfr
        # part_group_0.create_dataset('u', data=constructed_particles['internal_energy'].to_numpy(dtype=np.float64))                     # m^2 s^-2
        # part_group_0.create_dataset('nh', data=constructed_particles['neutral_hydrogen_fraction'].to_numpy(dtype=np.float64))        # dimensionless
        # part_group_0.create_dataset('ne', data=constructed_particles['electron_abundance'].to_numpy(dtype=np.float64))               # dimensionless
        part_group_0.create_dataset('T', data=constructed_particles['temperature'].to_numpy(dtype=np.float64))                         # K
        part_group_0.create_dataset('hden', data=constructed_particles['hden'].to_numpy(dtype=np.float64))                             # cm^-3
        # part_group_0.create_dataset('isrf_stellarBins', data=isrf_stellarBins)                                            # Habing units
        # part_group_0.create_dataset('isrf_skirt', data=constructed_particles['isrf'].to_numpy(dtype=np.float64))                       # Habing units
        part_group_0.create_dataset('Nh', data=constructed_particles['Nh'].to_numpy(dtype=np.float64))                                 # cm^-2

        # Metallicity: 
        part_group_0.create_dataset('z', data=metal_fractions_array)  # Mass ratio


    print(f"HDF5 file written successfully to: {write_file_path}")


    return None    

if __name__ == "__main__":
    main()