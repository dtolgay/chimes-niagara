import sys
sys.path.append("/scratch/m/murray/dtolgay")
import numpy as np 
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt 

from itertools import product

from scipy.optimize import curve_fit


from tools import constants

from concurrent.futures import ProcessPoolExecutor

import seaborn as sns 
sns.set_theme(style="ticks")


# Constants 
kpc2m = constants.kpc2m # kpc -> m
m2cm = constants.m2cm # meter -> cm

kb = constants.kb # J/K  --- Boltzmann Constant
gravitational_constant = constants.gravitational_constant # N m^2 kg^-2


def main(max_workers, spaxel_size, overlap_shift):

    directory_name = "voronoi_1e6"
    tasks = []

    redshift = "0.0"

    ###################### Firebox 
    galaxy_type = "firebox"
    
    # Define galaxy numbers to process
    galaxy_numbers = range(0, 1000)

    galaxy_names = [f"gal{num}" for num in galaxy_numbers]
    tasks_firebox = [(directory_name, galaxy_type, redshift, galaxy_name, spaxel_size, overlap_shift) for galaxy_name in galaxy_names]       
    tasks.append(tasks_firebox)


    ###################### Zoom-in 
    galaxy_type = "zoom_in"

    galaxy_names=[
        "m12b_res7100_md",
        "m12c_res7100_md",
        "m12f_res7100_md",
        "m12i_res7100_md",
        "m12m_res7100_md",
        "m12q_res7100_md",
        "m12r_res7100_md",
        "m12w_res7100_md",
        "m11d_r7100",
        "m11e_r7100",
        "m11h_r7100",
        "m11i_r7100",
        "m11q_r7100",    
    ]

    tasks_zoom_in = [(directory_name, galaxy_type, redshift, galaxy_name, spaxel_size, overlap_shift) for galaxy_name in galaxy_names]
    tasks.append(tasks_zoom_in)

#  ###################### particle_split 
#     galaxy_type = "particle_split"

#     galaxy_names=[
#         "m12i_r880_md",
#     ]

#     tasks_particle_split = [(directory_name, galaxy_type, redshift, galaxy_name, spaxel_size, overlap_shift) for galaxy_name in galaxy_names]
#     tasks.append(tasks_particle_split)    



    ### Run the jobs 
    tasks = sum(tasks, []) ## Flatten the list 
    print(f"tasks: {tasks}")

    # Use a process pool to execute tasks concurrently
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_galaxy, tasks))

    # Separate data_df and column_names_units
    data_dfs, column_names_units_list = zip(*results)
    column_names_units = column_names_units_list[0] # All of the elements are the same. 

    # Concatenate all DataFrames obtained from the processes
    spaxel_galaxies = pd.concat(data_dfs, ignore_index=True)

    # Write to a file 
    header = [
        f"{key} [{column_names_units[key]}]"
        for key in column_names_units
    ]

    save_dir = "/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/python_files/analyze_hden_metallicity_turbulence_isrf_radius/data"
    out_file_name = f"{save_dir}/spaxels_{directory_name}_spaxelSize{spaxel_size}_overlapShift{overlap_shift}_z0_usingIvalues_smoothingLength_hybridInterpolator.csv"
    spaxel_galaxies.to_csv(out_file_name, mode='w', sep=',', header=header, index=False, float_format='%.10e')
    print(f"\n\n File written to: {out_file_name}")

    return 0

## Functions required to parallelize.
def process_galaxy(params):
    try:
        # Assuming single_galaxy_calculation is defined elsewhere and imported
        data_df, column_names_units = single_galaxy_calculation(*params)
        return data_df, column_names_units
    except Exception as error:
        galaxy_name = params[3]
        print(f"Error in processing {galaxy_name}: {error}")
        return pd.DataFrame(), [] 

## Functions defined here 
def single_galaxy_calculation(directory_name, galaxy_type, redshift, galaxy_name, spaxel_size, overlap_shift):

    print(f"----------------------------------------------------------- {galaxy_name} -----------------------------------------------------------")

    base_dir = "/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius"
    fdir = f"{base_dir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}"

    # Read gas and star particles
    gas_particles_df, star_particles_df, line_names = read_star_and_gas_particles(fdir)

    # Create spaxels 
    # Consider only 20 kpc from the center of the galaxy 
    R_max = 20e3 #kpc

    R_gal_gas = np.sqrt(
        gas_particles_df["x"]**2 + gas_particles_df["y"]**2 + gas_particles_df["z"]**2 
    )

    R_gal_star = np.sqrt(
        star_particles_df["x"]**2 + star_particles_df["y"]**2 + star_particles_df["z"]**2 
    )

    indices_gas = np.where(R_gal_gas < R_max)[0]
    indices_star = np.where(R_gal_star < R_max)[0]


    gas_particles_df = gas_particles_df.iloc[indices_gas].copy().reset_index()
    star_particles_df = star_particles_df.iloc[indices_star].copy().reset_index()

    # Since gas_particles_df has index column new created column by reset_index is named as level_0
    gas_particles_df.drop(columns=['level_0'], inplace=True)  
    star_particles_df.drop(columns=['index'], inplace=True)


    # Match the gas and star particles with the boxes according to their position.
    max_length = R_max  # pc 

    print("Matching for gas particles started.")
    gas_indices, lower_left_corner_coordinates, upper_right_corner_coordinates, box_coordinates = find_indices_inside_boxes(
        max_length = max_length, 
        resolution = spaxel_size, 
        particles_df = gas_particles_df,
        overlap_shift=overlap_shift,
    )

    print("Matching for star particles started.")
    star_indices = find_indices_inside_boxes(
        max_length = max_length, 
        resolution = spaxel_size, 
        particles_df = star_particles_df,
    )[0]

    # ## Comment below to cancel plotting. TODO: 
    # print("Plotting.")
    # plot_spaxel(gas_particles_df, gas_indices, R_max)

    print("Calculating the properties of spaxels.")
    data_df, column_names_units = calculate_spaxel_properties(
            gas_particles_df, star_particles_df, spaxel_size, lower_left_corner_coordinates, gas_indices, star_indices, galaxy_name, redshift, line_names
        )

    return data_df, column_names_units

def plot_spaxel(gas_particles_df, gas_indices, R_max):
    ################################### Plotting

    index = 3000

    plt.hist2d(
            x=gas_particles_df["x"],
            y=gas_particles_df["y"],
            bins=500,
            norm=matplotlib.colors.LogNorm(),
            range=[[-R_max, R_max], [-R_max, R_max]]
        )
    plt.grid(True)
    plt.show()

    plt.hist2d(
            x=gas_particles_df.iloc[gas_indices[index]]["x"],
            y=gas_particles_df.iloc[gas_indices[index]]["y"],
            bins=500,
            norm=matplotlib.colors.LogNorm(),
            range=[[-R_max, R_max], [-R_max, R_max]]
        )
    plt.grid(True)
    plt.show()


    plt.hist2d(
            x=gas_particles_df.iloc[gas_indices[index-1]]["x"],
            y=gas_particles_df.iloc[gas_indices[index-1]]["y"],
            bins=500,
            norm=matplotlib.colors.LogNorm(),
            range=[[-R_max, R_max], [-R_max, R_max]]
        )
    plt.grid(True)
    plt.show()

    return 0

def read_star_and_gas_particles(fdir):

    ### Read gas and star particles
    # Read gas particles

    line_names = [
        "L_ly_alpha",  # [erg s^-1]
        "L_h_alpha", # [erg s^-1]
        "L_h_beta", # [erg s^-1]
        "L_co_10", # [K km s^-1 pc^2] 
        "L_co_21", # [K km s^-1 pc^2] 
        "L_co_32", # [K km s^-1 pc^2] 
        "L_co_43", # [K km s^-1 pc^2] 
        "L_co_54", # [K km s^-1 pc^2] 
        "L_co_65", # [K km s^-1 pc^2] 
        "L_co_76", # [K km s^-1 pc^2] 
        "L_co_87", # [K km s^-1 pc^2] 
        "L_13co",  # [K km s^-1 pc^2] 
        "L_c2", # [erg s^-1]
        "L_o3_88", # [erg s^-1]
        "L_o3_5006", # [erg s^-1]
        "L_o3_4958", # [erg s^-1]         
    ]

    gas_column_names = [
        "x",  # pc
        "y", 
        "z", 
        "smoothing_length", # pc 
        "mass",  # Msolar
        "metallicity",  # 
        "temperature", 
        "vx", 
        "vy", 
        "vz", 
        "hden", 
        "radius", 
        "sfr", 
        "turbulence", 
        "density", 
        "mu_theoretical",
        "average_sobolev_smoothingLength",
        "index", 
        "isrf",
    ] + line_names

    gas = np.loadtxt(
        fname=f"{fdir}/L_line_smoothingLength_hybridInterpolator_flux2Luminosity.txt",
        skiprows=1
    )

    gas = pd.DataFrame(gas, columns=gas_column_names)

    # Read star particles 
    star_column_names = [
        "x",  # pc
        "y",  
        "z",  
        "vx", # km/s
        "vy",
        "vz",
        "metallicity", # (1)
        "mass", # Msolar
        "age",  # Myr
    ]

    star = np.loadtxt(
        fname=f"{fdir}/comprehensive_star.txt",
        skiprows=1
    )

    star_particles_df = pd.DataFrame(star, columns=star_column_names)

    # Read semi_analytical_average_sobolev_smoothingLength.txt
    semi_analytical_column_names = [
        "x",
        "y",
        "z",
        "smoothing_length",
        "mass", 
        "metallicity",
        "temperature",
        "vx",
        "vy",
        "vz",
        "hden",
        "radius",
        "sfr",
        "turbulence",
        "density",
        "mu_theoretical",
        "average_sobolev_smoothingLength",
        "index",
        "isrf",
        "h2_mass",
        "Xco",
        "L_co_10_semi"
    ]

    semi_analytical = pd.DataFrame(
        np.loadtxt(f"{fdir}/semi_analytical_average_sobolev_smoothingLength.txt", skiprows = 1),
        columns=semi_analytical_column_names
    )

    gas_particles_df = gas.merge(semi_analytical[[
        "index",
        "h2_mass",
        "Xco",
        "L_co_10_semi"    
    ]], how='inner', on='index')


    return gas_particles_df, star_particles_df, line_names

def find_indices_inside_boxes(max_length, resolution, particles_df, overlap_shift=0):
    
    """
    Creating spaxels. These spaxels will overlap with each other. 
    """
    # Define x coordinates
    x = np.arange(-int(max_length), int(max_length), resolution)
    if overlap_shift != 0:
        shifted_x = x + overlap_shift
        x = np.sort(np.concatenate((x, shifted_x)))
    
    # Define y coordinates.
    y = x
    
    # The size of the box is the resolution
    lower_left_corner_coordinates = np.array(list(product(x, y)))
    upper_right_corner_coordinates = lower_left_corner_coordinates + resolution

    
    box_coordinates = []
    indices = []
    # Find the particles associated with each box 
    for i in range(len(lower_left_corner_coordinates)):

        # Compute boolean masks
        condition = (
            (particles_df["x"] >= lower_left_corner_coordinates[i][0]) &
            (particles_df["x"] < upper_right_corner_coordinates[i][0]) &
            (particles_df["y"] >= lower_left_corner_coordinates[i][1]) &
            (particles_df["y"] < upper_right_corner_coordinates[i][1])
        )


        # Use boolean indexing directly to filter rows
        filtered_df = particles_df[condition]

        indices.append(filtered_df.index.values) 
        
        
        box_coordinates.append([lower_left_corner_coordinates[i], upper_right_corner_coordinates[i]])
        
        if i % 1e3 == 1:
            print(f"{i} finished. Left {len(lower_left_corner_coordinates) - i}")
            
            
    box_coordinates = np.array(box_coordinates)
    print(f"Number of boxes: {len(box_coordinates)}")
    print("Done!")
        
    return indices, lower_left_corner_coordinates, upper_right_corner_coordinates, box_coordinates

def sfr_calculator(star_df: pd.DataFrame, within_how_many_Myr:float):
    indices = np.where(star_df["age"] <= within_how_many_Myr)[0]
    sfr_star = np.sum(star_df.iloc[indices]["mass"]) / (10 * 1e6)  # Msolar / year
    return sfr_star

# This is the used half light radius. Found by using the stellar mass. 
def find_half_light_radius_star_mass(star_particles_df):

    R_star = np.sqrt(star_particles_df['x']**2 + star_particles_df['y']**2 + star_particles_df['z']**2)

    # Define some constants to use them later
    R_max = max(R_star)
    number_of_bins = int(1e3) + 1 # Number of annulus is number_of_bins - 1 
    total_star_mass = sum(star_particles_df['mass'])

    # There is Rmax/number_of_bins distance between two radius that divides annulus from each other
    radius_boundaries_for_bins = np.linspace(0, R_max, number_of_bins) 
    outer_radius_for_bins = radius_boundaries_for_bins[1:]                                                  # pc
    inner_radius_for_bins = radius_boundaries_for_bins[:-1]                                                 # pc

    # Categorization will be done in here. I will use numpy.digitize to find the matching between star and created annuli.
    digitize_gas_bins = np.digitize(R_star, radius_boundaries_for_bins)

    summed_star_mass = 0 
    for i in range(max(digitize_gas_bins)):
        summed_star_mass += np.sum(star_particles_df.loc[digitize_gas_bins == i, 'mass'])
        if (summed_star_mass * 2 > total_star_mass): 
            break 

    return (outer_radius_for_bins[i] + inner_radius_for_bins[i]) / 2 

# # Calculate spaxel properties
# def calculate_spaxel_properties(gas_particles_df, star_particles_df, resolution, lower_left_corner_coordinates, gas_indices, star_indices, galaxy_name, redshift, line_names):

#     # Calculate half light radius. 
#     half_light_radius = find_half_light_radius_star_mass(star_particles_df=star_particles_df)

#     # Calculate the properties inside the boxes

#     area = resolution**2 # pc^2

#     # Surface densities

#     surface_density_per_kpc_square = []
#     pressure_terms = []

#     for i in range(len(lower_left_corner_coordinates)):

#         center_x = lower_left_corner_coordinates[i][0] + resolution/2
#         center_y = lower_left_corner_coordinates[i][1] + resolution/2

#         # Calculate the distance 
#         distance = np.sqrt(center_x ** 2 + center_y ** 2)
        
#         filtered_gas_particles = gas_particles_df.iloc[gas_indices[i]].copy()
#         filtered_star_particles = star_particles_df.iloc[star_indices[i]].copy()
            
#         # Star mass
#         sigma_star_mass = sum(filtered_star_particles["mass"]) / (area*1e-6) # Msolar / kpc^2
        
#         # SFR 
#         sigma_sfr = sum(filtered_gas_particles["sfr"]) / (area*1e-6) # Msolar / year / kpc^2
#         sigma_sfr_10Myr = sfr_calculator(star_df = filtered_star_particles, within_how_many_Myr = 10) / (area*1e-6) # Msolar / year / kpc^2
        
#         # All gas mass
#         sigma_all_gas = sum(filtered_gas_particles["mass"]) / (area*1e-6) # Msolar / kpc^2
        
#         # Molecular gas mass
#         sigma_h2 = sum(filtered_gas_particles["h2_mass"]) / (area*1e-6) # Msolar / kpc^2
        
#         # Line Luminosities
#         surface_density_lines = {} 
#         for line in line_names: 
#             surface_density_lines[line] = sum(filtered_gas_particles[line]) / (area*1e-6) # Unit of Luminosity / kpc^2 

#         # TODO: Delete
#         # sigma_L_ly_alpha = sum(filtered_gas_particles["L_ly_alpha"]) / (area*1e-6) # erg s^-1 / kpc^2
#         # sigma_L_h_alpha = sum(filtered_gas_particles["L_h_alpha"]) / (area*1e-6) # erg s^-1 / kpc^2
#         # sigma_L_h_beta = sum(filtered_gas_particles["L_h_beta"]) / (area*1e-6) # erg s^-1 / kpc^2
#         # sigma_L_co_10 = sum(filtered_gas_particles["L_co_10"]) / (area*1e-6) # K - km s^-1 pc^2 / kpc^2
#         # sigma_L_o3_5006 = sum(filtered_gas_particles["L_o3_5006"]) / (area*1e-6) # erg s^-1 / kpc^2
        
#         # Pressure 
#         # Using the eqn 2 in Ellison 2024 The almaquest survey paper: https://ui.adsabs.harvard.edu/abs/2024MNRAS.52710201E/abstract
#         R_star = half_light_radius * 1e-3 / 1.68   # kpc
#         rho_star = sigma_star_mass / (0.54 * R_star) # Msolar / kpc^3
#         gas_velocity_dispersion = 11 # km / s
        
#         sigma_all_gas_unit_converted = sigma_all_gas * constants.Msolar2kg / kpc2m**2  # kg / m^2
#         rho_star_unit_converted = rho_star * constants.Msolar2kg / kpc2m**3            # kg / m^3
#         gas_velocity_dispersion_unit_converted = gas_velocity_dispersion * 1e3         # m / s
        
#     #     pressure = np.pi * gravitational_constant * sigma_all_gas_unit_converted / 2 + \
#     #     sigma_all_gas_unit_converted * np.sqrt(2 * gravitational_constant * rho_star_unit_converted) * gas_velocity_dispersion_unit_converted
        
#         pressure_gas = np.pi * gravitational_constant * sigma_all_gas_unit_converted**2 / 2
        
#         pressure_star = sigma_all_gas_unit_converted * np.sqrt(2 * gravitational_constant * rho_star_unit_converted) *\
#         gas_velocity_dispersion_unit_converted
        
#         pressure = pressure_gas + pressure_star
        
#         pressure_over_kb = pressure / kb / m2cm**3  # K / cm^3
        

#         pressure_terms.append([pressure_gas / kb / m2cm**3, pressure_star / kb / m2cm**3])
        
#         surface_density_per_kpc_square.append(
#         np.append(
#         [
#             sigma_star_mass,            # Msolar / kpc^2
#             sigma_sfr,                  # Msolar / year / kpc^2
#             sigma_sfr_10Myr,            # Msolar / year / kpc^2
#             sigma_all_gas,              # Msolar / kpc^2
#             sigma_h2,                   # Msolar / kpc^2
#             pressure_over_kb,           # K / cm^3
#         ], list(surface_density_lines.values())
#         ))
        
#     line_names_with_sigma = [f"sigma_{line}" for line in line_names]
        
#     column_names = [
#         "sigma_star_mass",
#         "sigma_sfr",
#         "sigma_sfr_10Myr",
#         "sigma_all_gas",
#         "sigma_h2",
#         "pressure_over_kb",
#     ] + line_names_with_sigma

#     surface_density_per_kpc_square = pd.DataFrame(surface_density_per_kpc_square, columns=column_names)

#     pressure_terms = pd.DataFrame(pressure_terms, columns=["gas", "star"])


#     # Derive other spaxel properties that does not require to do in the for loop
#     # surface_density_per_kpc_square["depletion_time"] = surface_density_per_kpc_square["sigma_h2"] / surface_density_per_kpc_square["sigma_sfr"]
#     surface_density_per_kpc_square["depletion_time"] = surface_density_per_kpc_square["sigma_h2"] / surface_density_per_kpc_square["sigma_sfr_10Myr"]



#     # Copy surface_density_per_kpc_square dataframe to data_df
#     data_df = pd.DataFrame()

#     # Copy pressure_terms dataframe to data_df
#     for column in pressure_terms.keys():
#         data_df[f"log_{column}_pressure_over_kb"] = np.log10(pressure_terms[column]).copy()


#     for column in surface_density_per_kpc_square.keys():
#         data_df[f"log_{column}"] = np.log10(surface_density_per_kpc_square[column])

        
#     data_df["distance"] = distance
#     data_df["galaxy_name"] = galaxy_name
#     data_df["redshift"] = redshift

#     data_df["x_center"] = center_x
#     data_df["y_center"] = center_y
#     data_df["spaxel_size"] = resolution


#     # Get rid of the inf and nan values inside the dataframe
#     # data_df = data_df[~data_df.isin([np.inf, -np.inf, np.nan]).any(axis=1)] 

#     return data_df

# Calculate spaxel properties
def calculate_spaxel_properties(
        gas_particles_df, 
        star_particles_df, 
        resolution, 
        lower_left_corner_coordinates, 
        gas_indices, 
        star_indices, 
        galaxy_name, 
        redshift, 
        line_names,
    ):

    # Calculate half light radius. 
    half_light_radius = find_half_light_radius_star_mass(star_particles_df=star_particles_df)

    # Calculate the properties inside the boxes

    area = resolution**2 # pc^2

    values_for_all_spaxels = []
    
    for i in range(len(lower_left_corner_coordinates)):
        calculated_properties = {} # Fresh start in each iteration.

        x_center = lower_left_corner_coordinates[i][0] + resolution/2   # pc
        y_center = lower_left_corner_coordinates[i][1] + resolution/2   # pc 

        # Calculate the distance 
        distance = np.sqrt(x_center ** 2 + y_center ** 2)               # pc 
        calculated_properties.update({
            "x_center": {
                "unit": "pc",
                "value": x_center,
                "column_name": "x_center",
            },
            "y_center": {
                "unit": "pc",
                "value": y_center,
                "column_name": "y_center",
            },
            "distance": {
                "unit": "pc",
                "value": distance,
                "column_name": "distance",
            },                        
        })        
        

        # Filter gas particles so that you only left with the gas particles inside the spaxel.
        filtered_gas_particles = gas_particles_df.iloc[gas_indices[i]].copy()
        filtered_star_particles = star_particles_df.iloc[star_indices[i]].copy()

        # Calculate the area in terms of kpc^2
        area_in_kpc_square = area*1e-6 # kpc^2
            
        # Star mass
        sigma_star_mass = sum(filtered_star_particles["mass"]) / area_in_kpc_square # Msolar / kpc^2
        calculated_properties.update({
            "sigma_star_mass": {
                "unit": "Msolar / kpc^2",
                "value": sigma_star_mass,
                "column_name": "sigma_star_mass",
            }
        })
        
        # SFR 
        sigma_sfr = sum(filtered_gas_particles["sfr"]) / area_in_kpc_square # Msolar / year / kpc^2
        sigma_sfr_10Myr = sfr_calculator(star_df = filtered_star_particles, within_how_many_Myr = 10) / area_in_kpc_square # Msolar / year / kpc^2
        calculated_properties.update({
            "sigma_sfr": {
                "unit": "Msolar / year / kpc^2",
                "value": sigma_sfr,
                "column_name": "sigma_sfr",
            },
            "sigma_sfr_10Myr": {
                "unit": "Msolar / year / kpc^2",
                "value": sigma_sfr_10Myr,
                "column_name": "sigma_sfr_10Myr",
            },            
        })
        
        # All gas mass
        sigma_all_gas = sum(filtered_gas_particles["mass"]) / area_in_kpc_square # Msolar / kpc^2
        calculated_properties.update({
            "sigma_all_gas": {
                "unit": "Msolar / kpc^2",
                "value": sigma_all_gas,
                "column_name": "sigma_all_gas",
            },
        })        
        
        # Molecular gas mass
        sigma_h2 = sum(filtered_gas_particles["h2_mass"]) / area_in_kpc_square # Msolar / kpc^2
        calculated_properties.update({
            "sigma_h2": {
                "unit": "Msolar / kpc^2",
                "value": sigma_h2,
                "column_name": "sigma_h2",
            },
        })        
                
        
        # Line Luminosities
        observer_CO_unit_CO_lines = [
            "L_co_10",
            "L_co_21",
            "L_co_32",
            "L_co_43",
            "L_co_54",
            "L_co_65",
            "L_co_76",
            "L_co_87",
            "L_13co",
        ]
        surface_density_lines = {} 
        for line in line_names: 
            surface_density_lines[line] = sum(filtered_gas_particles[line]) / area_in_kpc_square # Unit of Luminosity / kpc^2 
            unit = "K km s^-1 pc^2 / kpc^2" if line in observer_CO_unit_CO_lines else "erg s^-1 / kpc^2"
            calculated_properties.update({
                f"sigma_{line}": {
                    "unit": unit,
                    "value": surface_density_lines[line],
                    "column_name": line,
                },
            })                

        
        # Pressure 
        # Using the eqn 2 in Ellison 2024 The almaquest survey paper: https://ui.adsabs.harvard.edu/abs/2024MNRAS.52710201E/abstract
        R_star = half_light_radius * constants.pc2kpc / 1.68   # kpc
        rho_star = sigma_star_mass / (0.54 * R_star) # Msolar / kpc^3
        gas_velocity_dispersion = 11 # km / s
        
        sigma_all_gas_SI = sigma_all_gas * constants.Msolar2kg / kpc2m**2  # kg / m^2
        rho_star_SI = rho_star * constants.Msolar2kg / kpc2m**3            # kg / m^3
        gas_velocity_dispersion_SI = gas_velocity_dispersion * 1e3         # m / s

        # Calculating gas pressure         
        pressure_gas = np.pi * gravitational_constant * sigma_all_gas_SI**2 / 2     # N / m^2
        pressure_gas_over_kb = pressure_gas / kb / m2cm**3          # K / cm^3 
        calculated_properties.update({
            f"pressure_gas_over_kb": {
                "unit": "K / cm^3",
                "value": pressure_gas_over_kb,
                "column_name": "pressure_gas_over_kb",
            },
        })          
    
        pressure_star = sigma_all_gas_SI * np.sqrt(2 * gravitational_constant * rho_star_SI) * gas_velocity_dispersion_SI   # N / m^2
        pressure_star_over_kb = pressure_star / kb / m2cm**3        # K / cm^3 
        calculated_properties.update({
            f"pressure_star_over_kb": {
                "unit": "K / cm^3",
                "value": pressure_star_over_kb,
                "column_name": "pressure_star_over_kb",
            },
        })            
        
        
        pressure_over_kb = pressure_gas_over_kb + pressure_star_over_kb  # K / cm^3
        calculated_properties.update({
            f"pressure_over_kb": {
                "unit": "K / cm^3",
                "value": pressure_over_kb,
                "column_name": "pressure_over_kb",
            },
        })                    


        # I calculated the spaxel properties. Now I will add it into the dataframe
        # Start a new arrays in after each spaxel.        
        values_for_this_run = []     
        column_names = [] 
        units = []   
        for key in list(calculated_properties.keys()):
            values_for_this_run.append(calculated_properties[key]["value"])
            column_names.append(f'{calculated_properties[key]["column_name"]}')
            units.append(f'{calculated_properties[key]["unit"]}')

        values_for_all_spaxels.append(values_for_this_run)
        
        
        
    ##### Outside of the loop 
    values_for_all_spaxels = np.array(values_for_all_spaxels)
    data_df = pd.DataFrame(values_for_all_spaxels, columns=column_names)
            
        
    # Derive other spaxel properties that does not require to do in the for loop
    data_df["depletion_time"] = data_df["sigma_h2"] / data_df["sigma_sfr_10Myr"]
    column_names.append("depletion_time")
    units.append("1 / year")

        
    data_df["galaxy_name"] = galaxy_name
    column_names.append("galaxy_name")
    units.append("1")

    data_df["redshift"] = redshift
    column_names.append("redshift")
    units.append("1")
    
    data_df["spaxel_length"] = resolution
    column_names.append("spaxel_length")
    units.append("pc")

    column_names_units = dict(zip(column_names, units))


    return data_df, column_names_units    

#### 

if __name__ == "__main__":
    max_workers = 10 # TODO
    overlap_shift = 0 # No overlap between the spaxels 
    # spaxel_size = 1e3 # pc
    spaxel_size = 1e4 # pc



    main(max_workers, spaxel_size, overlap_shift)