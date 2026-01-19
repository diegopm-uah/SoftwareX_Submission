import pandas as pd
from pathlib import Path
import os
import math
import stl
from datetime import datetime
import argparse
import subprocess
import time
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import shutil

seed = 53
seed = datetime.now().microsecond

def fill_stl():
    """
    Reads a CSV file containing STL file metadata, identifies rows with missing values 
    in specific columns ('facets', 'bbx', 'bby', 'bbz'), and fills in the missing data 
    by analyzing the corresponding STL files. The function calculates the number of facets 
    and the bounding box dimensions (x, y, z) for each STL file and updates the CSV file 
    with the computed values.
    The updated data is saved back to the same CSV file.
    Raises:
        FileNotFoundError: If the CSV file or any referenced STL file is not found.
        ValueError: If the CSV file has an unexpected format or missing required columns.
    """

    df = pd.read_csv(Path("data/stl.csv"), sep=";", index_col="name")
    
    df_with_nan = df[pd.isna(df[['facets','bbx','bby','bbz']]).any(axis=1)]
    
    for name in df_with_nan.index:
        mesh = stl.mesh.Mesh.from_file(Path("data/stl/" + name + ".stl"))
        
        df.loc[name, 'facets'] = len(mesh.vectors)
        df.loc[name, 'bbx'] = mesh.x.max() - mesh.x.min()
        df.loc[name, 'bby'] = mesh.y.max() - mesh.y.min()
        df.loc[name, 'bbz'] = mesh.z.max() - mesh.z.min()

    df.reset_index().to_csv(Path("data/stl.csv"), sep=';', header=True, index=False)
    
def change_num_facets(geometries: list[str], target_faces: int):
    """
    Adjusts the number of facets (faces) in a list of 3D geometry files to match a target number of faces.
    This function takes a list of geometry file names (without extensions) and modifies the number of 
    faces in their corresponding STL files to approximate the specified target number of faces. The 
    modified STL files are saved with a new name indicating the target face count.
    Args:
        geometries (list[str]): A list of geometry file names (without the ".stl" extension) to process.
        target_faces (int): The desired number of faces for the output geometries.
    Notes:
        - The function assumes the STL files are located in the "data/stl" directory relative to the 
          script's location.
        - Blender's Python API is used to perform the operations, so Blender must be installed and 
          properly configured.
        - The function requires the "STL format (legacy)" extension to be installed in Blender for 
          importing and exporting STL files.
        - The decimation ratio is calculated to approximate the target number of faces, but the exact 
          number of faces in the output may vary.
    """

    import bpy
    import bmesh

    geom_folder_path = Path("data/stl")

    for geometry in geometries:

        input_filepath = geom_folder_path / f"{geometry}.stl"
        output_filepath = geom_folder_path / f"{geometry}_{target_faces}.stl"

        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()
        bpy.ops.wm.stl_import(filepath=str(input_filepath)) # Imports of STL vary with the Blender version
        obj = bpy.context.selected_objects[0]

        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        current_faces = len(bm.faces)
        bm.free()
        
        ratio = math.ceil(target_faces / current_faces * 10000) / 10000

        decimate_modifier = obj.modifiers.new(name='Decimate', type='DECIMATE')
        decimate_modifier.ratio = ratio

        bpy.ops.object.make_single_user(object=True, obdata=True, material=False, animation=False)

        bpy.ops.object.modifier_apply(modifier='Decimate') # Needs the extension "STL format (legacy)" to be installed manually on Blender

        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        print(f'\nGeometry {geometry}: from {current_faces} faces to {len(bm.faces)} faces (target was {target_faces})\n')
        bm.free()

        bpy.ops.export_mesh.stl(filepath=str(output_filepath)) # Exports of STL vary with the Blender version

def add_stl(stl_name: str, category: str):
    """
    Adds a new STL file entry to the 'data/stl.csv' file.
    This function checks if the specified STL file exists in the 'data/stl/' directory
    and ensures that the STL name is not already present in the CSV file. If both conditions
    are met, it appends the new STL entry with its associated category to the CSV file.
    Args:
        stl_name (str): The name of the STL file (without the '.stl' extension).
        category (str): The category to associate with the STL file.
    Raises:
        AssertionError: If the STL file does not exist in the 'data/stl/' directory.
        AssertionError: If the STL name already exists in the 'data/stl.csv' file.
    """

    df = pd.read_csv(Path("data/stl.csv"), sep=";")
    
    assert os.path.isfile(Path('data/stl/' + stl_name + '.stl')), f'El STL a añadir {stl_name} no existe'
    assert len(df[df['name']==stl_name]) == 0, f'El STL {stl_name} ya se ha introducido'
    
    final_df = pd.concat([df, pd.DataFrame({'name':[stl_name], 'category':[category]})])
    final_df.to_csv(Path("data/stl.csv"), sep=';', header=True, index=False)

def remove_stl(stl_name: str):
    """
    Removes a specific STL entry from the 'data/stl.csv' file.
    This function reads the 'data/stl.csv' file into a pandas DataFrame, 
    verifies that the specified STL name exists in the file, and removes 
    the corresponding entry. The updated DataFrame is then saved back 
    to the same file.
    Args:
        stl_name (str): The name of the STL entry to be removed.
    Raises:
        AssertionError: If the specified STL name does not exist in the CSV file.
    Note:
        The CSV file is expected to have a column named 'name' as the index 
        and use a semicolon (';') as the delimiter.
    """

    df = pd.read_csv(Path("data/stl.csv"), sep=";", index_col='name')
    
    assert len(df.filter(items=[stl_name], axis='index')) == 1, f'El STL {stl_name} no existe en el CSV'
    
    df.drop(index=stl_name, axis='index', inplace=True)
    df.reset_index().to_csv(Path("data/stl.csv"), sep=';', header=True, index=False)

# Create a new case in the CSV param, creating the new folders as appropriate.
def create_case(case: str,
                delta_angle: float, central_freq: float, delta_freq: float,
                num_angle: int, num_freq: int):
    
    assert case in ('theta', 'phi'), "The case variable can only be 'theta' or 'phi'."
    
    # Creation of the new CSV cases
    df_cases = pd.read_csv(Path("data/param.csv"), index_col=0, sep=";")
    new_df = pd.concat([df_cases,
                        pd.DataFrame([[case, delta_angle, central_freq, delta_freq, num_angle, num_freq]],
                                     columns=df_cases.columns)],
                       ignore_index=True)
    last_index = new_df.last_valid_index()
    new_df.to_csv(Path("data/param.csv"), index=True, sep=';')
    
    # Creation of the corresponding folders
    df_stl = pd.read_csv(Path("data/stl.csv"), sep=";")
    geom_names = df_stl['name']
    path_case = Path("data/cases/") / str(last_index)
    os.mkdir(path_case)
    for name in geom_names:
        os.mkdir(path_case / name)
        pd.DataFrame(
            {'theta': [], 'phi': [], 'file': []}
                     ).to_csv(path_case / name / "rcs.csv", index=False, sep=';')
        os.mkdir(path_case / name / "rcs")
        
def generate_frequency_file(central_freq, freq_width, m):
    """Generates a text file with frequency values centered around a given central frequency.

    Args:
        central_freq (float): The central frequency value.
        freq_width (float): The range of variation around the central frequency. Half of the total range.
        m (int): Number of frequencies to generate.
        filename (str, optional): Output file name.
    """
    filename = Path("data/_out/frequencies.txt")
    with open(filename, "w") as f:
        freq_values = np.linspace(central_freq - freq_width, central_freq + freq_width, m)
        for _, freq in enumerate(freq_values):
            f.write(f"{freq}\n")
            
def cartesian_to_spherical(x, y, z):
    """Converts Cartesian coordinates to spherical coordinates.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
        z (float): Z coordinate.

    Returns:
        tuple: (theta, phi) where theta is in (0, pi) and phi is in (0, 2pi).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # Polar angle (0 to pi)
    phi = np.arctan2(y, x)    # Azimuthal angle (0 to 2pi)
    if phi < 0:
        phi += 2 * np.pi  # Ensure phi is in (0, 2pi)
    return theta, phi
            
def generate_spherical_coordinates_file(n, angle, width, m,  filename, cone_width=None, pov=None, theta_limits=None, phi_limits=None):
    """Generates a text file with n samples of random spherical coordinates and additional variations.

    Each sample consists of a random unit direction in spherical coordinates. Additional directions 
    are generated by modifying either the theta or phi value within a specified width.

    Args:
        n (int): Number of samples to generate.
        angle (str): Angle to modify ("theta" or "phi").
        width (float): Range of variation in degrees for the selected angle.
        m (int): Number of directions to generate per sample.
        filename (str, optional): Output file name.
        pov (str): Which part of the object will be being looked.
        cone_width (float): Width of the cone of view when pov is present.
        theta_limits (tuple): Optional (min, max) in degrees for theta.
        phi_limits (tuple): Optional (min, max) in degrees for phi.
    """

    def pov_angles(pov, cw):
        if pov == 'top':
            return cw * np.random.rand() * np.pi/180, np.random.rand() * 2 *  np.pi
        elif pov == 'bottom':
            return np.pi - cw * np.random.rand() * np.pi/180, np.random.rand() * 2 *  np.pi
        else:
            povs = {"front": (90, 0), "left_side": (90, 90), "right_side": (90, 270), "back": (90, 180)}
            general = cw * (np.random.rand() * 2 - 1)
            general_2 = cw * (np.random.rand() * 2 - 1)
            return (general + povs[pov][0]) * np.pi/180, (general_2 + povs[pov][1]) * np.pi/180
    
    def sample_bounded_angles(t_lim, p_lim):
        # Samples uniformly between min and max degrees, converts to radians
        t = np.random.uniform(np.radians(t_lim[0]), np.radians(t_lim[1]))
        p = np.random.uniform(np.radians(p_lim[0]), np.radians(p_lim[1]))
        return t, p

    width = np.radians(width)  # Convert width from degrees to radians
    ret_values = list()
    
    with open(filename, "w") as f:
        for _ in range(n):
            if theta_limits is not None and phi_limits is not None:
                theta, phi = sample_bounded_angles(theta_limits, phi_limits)
            elif pov:  # Positive x axis
                theta, phi = pov_angles(pov, cone_width)
            else:
                # 1. Generate uniform random variables
                u = np.random.rand()
                v = np.random.rand()

                # 2. Calculate angles
                theta = 2 * np.pi * u
                phi = np.arccos(1 - 2 * v)
                
            ret_values.append((theta,phi))
            
            # Generate additional directions centered on theta|phi
            if angle == "theta":
                values = np.linspace(theta - width, theta + width, m)
                for t in values:
                    f.write(f"{t}|{phi}\n")
            elif angle == "phi":
                values = np.linspace(phi - width, phi + width, m) 
                for p in values:
                    f.write(f"{theta}|{p}\n")
    return ret_values

def to_degrees(filename, filedest):
    with open(filename, "r") as f:
        with open(filedest, "w") as fdest:
            for line in f:
                print("|".join(list(map(lambda x: str(float(x) * 180.0 / math.pi), line.split("|")))), file=fdest)

def get_pov_intervals(pov, cw):
    """
    Returns positive angular intervals (min, max).
    Coordinate System: Theta [0, 180], Phi [0, 360).
    Front=0, Left=90, Back=180, Right=270.
    """
    centers = {
        "front":      (90, 0),    # The seam is here!
        "left_side":  (90, 90),
        "back":       (90, 180),  # Continuous range
        "right_side": (90, 270),  # Continuous range
        "top":        (0, 0),     # Pole
        "bottom":     (180, 0)    # Pole
    }
    
    if pov not in centers:
        raise ValueError(f"Unknown POV: {pov}")
        
    c_theta, c_phi = centers[pov]
    
    # --- THETA (Standard Clamping 0-180) ---
    t_min = max(0, c_theta - cw)
    t_max = min(180, c_theta + cw)
    theta_lims = (t_min, t_max)

    # --- PHI (Wrapping 0-360) ---
    if t_min == 0 or t_max == 180: # Poles
        phi_lims = None 
        print(f"[LOGIC] POV '{pov}' is polar. Selecting ALL Phi.")
    else:
        # Calculate raw mathematical limits
        p_min_raw = c_phi - cw
        p_max_raw = c_phi + cw
        
        # Normalize to 0-360
        p_min = p_min_raw % 360
        p_max = p_max_raw % 360
        
        phi_lims = (p_min, p_max)
        
        # Check for wrap-around, if the raw range crosses 0/360 (e.g. -20 to +20), p_min (340) will be > p_max (20)
        if p_min > p_max:
            print(f"[LOGIC] POV '{pov}' wraps 360. Range: {p_min:.1f}° -> 360° -> {p_max:.1f}°")
    
    return theta_lims, phi_lims

# Function to handle angle wrapping (for now, I'm leaving it unimplemented, but it would be useful if we don't want to include negative numbers in the intervals).
def filter_angle(df, column, limit_min, limit_max):
    """
    Filter a DataFrame based on angle values within a specified range.
    This function filters rows from a DataFrame where the values in a specified
    column fall within the given angle limits. It handles both standard ranges
    (e.g., 10 to 50 degrees) and wrap-around cases for circular angle data
    (e.g., 350 to 10 degrees, crossing the 0/360 boundary).
    Args:
        df: The pandas DataFrame to filter.
        column: The name of the column containing angle values to filter on.
        limit_min: The minimum angle value of the desired range.
        limit_max: The maximum angle value of the desired range.
    Returns:
        A filtered pandas DataFrame containing only rows where the angle values
        fall within the specified range.
    """

    if limit_min <= limit_max:
        # Standard case: e.g., 10 to 50
        return df[(df[column] >= limit_min) & (df[column] <= limit_max)]
    else:
        # Wrap-around case: e.g., 350 to 10, we want angles > 350 OR angles < 10
        return df[(df[column] >= limit_min) | (df[column] <= limit_max)]

# Main function. Creates new .npy files that store the RCS of various cases for various geometries.
def calculate_rcs(geometries: list[str], 
                  num_samples: int, 
                  case: int,
                  interval: tuple[str, float] | tuple[tuple, tuple] | None=None
                  ): 
    # Obtaining case parameters
    df_cases = pd.read_csv(Path("data/param.csv"), sep=";", index_col=0)
    sweep_angle, delta_angle, central_f, delta_f, num_angle, num_f = df_cases.loc[case,
        ['sweep_angle','delta_angle','central_f','delta_f','num_angle','num_f']
        ]
    
    generate_frequency_file(central_f, delta_f, num_f)
    directions_path = Path("data/_out/directions.txt")

    print("Generation has started...\n")
    
    from data.openRCS.rcs_monostatic import rcs_monostatic
    from data.openRCS.rcs_functions import extractCoordinatesData
    from data.openRCS.stl_module import stl_converter
    
    for geometry in geometries:

        print(f"---------------- Generating RCS samples for geometry: {geometry} ----------------\n")

        tic = time.time()

        stl_path = Path("data/stl") / (geometry + ".stl")
        
        if interval != None and type(interval[0]) == str:
            _ = generate_spherical_coordinates_file(num_samples, sweep_angle, delta_angle, num_angle,
                                                directions_path, cone_width=interval[1], pov=interval[0], theta_limits=None, phi_limits=None)
        elif interval != None and type(interval[0]) == list:
            _ = generate_spherical_coordinates_file(num_samples, sweep_angle, delta_angle, num_angle,
                                                directions_path, cone_width=None, pov=None, theta_limits=interval[0], phi_limits=interval[1])
        else:
            _ = generate_spherical_coordinates_file(num_samples, sweep_angle, delta_angle, num_angle,
                                                directions_path, cone_width=None, pov=None, theta_limits=None, phi_limits=None)
        
        to_degrees(Path("data/_out/directions.txt"), Path("data/_out/directions_degrees.txt"))

        stl_converter(file_path=stl_path)
        coordinatesData = extractCoordinatesData(0) # The parameter is resistivity. (PEC=0)
        
        ########## Creating files #############
        # These lines are for cases where the folder and the geometry CSV file do not yet exist, 
        # a situation that needs to be reviewed, since the create case only creates folders for geometries that already exist. 
        # If a new geometry is added, there is no way to create the folder without this step:
        folder_path = Path("data/cases") / str(case) / geometry / "rcs"
        os.makedirs(folder_path, exist_ok=True)
        if not os.path.isfile(Path("data/cases") / str(case) / geometry / "rcs.csv"):
            pd.DataFrame({'theta': [], 'phi': [], 'file': []}).to_csv(Path("data/cases") / str(case) / geometry / "rcs.csv", index=False, sep=';')
        
        # list_lines is the list of i-th lines of the files
        rcs_df = pd.read_csv(Path("data/cases") / str(case) / geometry / "rcs.csv", sep=";")
        data_list = list()
        max_file = rcs_df['file'].max() if len(rcs_df['file']) > 0 else 0     
        
        try:
            with open(Path("data/_out/frequencies.txt"), "r") as freq_file:
                freq_list = list(map(float, freq_file.readlines()))
            
            with open(Path("data/_out/directions_degrees.txt"), "r") as dir_file:
                dir_array_temp = dir_file.readlines()
                dir_array = [dir_array_temp[i*num_angle:(i+1)*num_angle] for i in range(len(dir_array_temp)//num_angle)]
                
            for directions in dir_array:
                rcs_matrix = np.zeros([num_angle, len(freq_list)], dtype=np.complex64)
                first_angle = np.array(list(map(float, directions[0].split("|"))))
                last_angle = np.array(list(map(float, directions[-1].split("|"))))
                step = (last_angle - first_angle) / (num_angle - 1) - 1e-7
            
                for i, freq in enumerate(freq_list):
                    params_entrys = [
                        stl_path,
                        freq,
                        0, 0, # correlation and std
                        1, # 1 if electric field, 2 if magnetic field
                        0, # Material, for PEC at zero
                        first_angle[1], last_angle[1], step[1] if step[1] > 0 else 0, # first phi, last phi, step phi
                        first_angle[0], last_angle[0], step[0] if step[0] > 0 else 0, # first theta, last theta, step theta
                    ]
                    
                    rcs_matrix[:, i] = np.array(rcs_monostatic(params_entrys, coordinatesData))

                max_file += 1
            
                # Creation of the new .txt file
                path_new_file = folder_path / (str(max_file) + ".txt")
                
                # We update rcs.csv
                data_list.append({'theta': (first_angle[0] + last_angle[0]) / 2,
                                    'phi': (first_angle[1] + last_angle[1]) / 2,
                                    'file': max_file})
                data_view = rcs_matrix.view(np.float32).reshape(rcs_matrix.shape[0], -1)
                formato_fila = ";".join(["%.3E %.3E"] * rcs_matrix.shape[1])

                np.savetxt(path_new_file, data_view, fmt=formato_fila)

            print(f"The Isar Aprox is equal to {num_angle}\n")
        except Exception as ex:
            raise ex
        
        toc_1 = time.time()
        print(f"OpenRCS generation of {num_samples} samples of the {geometry} completed in {toc_1-tic} seconds, {(toc_1-tic)/60} minutes.\n")
        
        os.remove(Path("coordinates.txt"))
        os.remove(Path("facets.txt"))
        
        rcs_df = pd.concat([rcs_df, pd.DataFrame(data_list)], ignore_index=True)
        rcs_df.to_csv(Path("data/cases") / str(case) / geometry / "rcs.csv", sep=";", header=True, index=False)

    print("\nGeneration of samples completed for all geometries.\n")
    
def npy_fill(case: int):
    """
    Scans geometry folders, infers matrix dimensions dynamically from .txt files, 
    and converts them to .npy only if they don't already exist.
    
    Args:
        case (int): Needed for the root directory (e.g., 'data/cases/0').
    """
    base_path = os.path.join("data", "cases", str(case))
    # 1. Find all geometry folders
    geometry_folders = [f for f in glob.glob(os.path.join(base_path, "*")) if os.path.isdir(f)]
    
    print(f"Scanning {len(geometry_folders)} geometries in {base_path}...")

    for geo_folder in geometry_folders:
        rcs_txt_folder = os.path.join(geo_folder, "rcs")
        rcs_npy_folder = os.path.join(geo_folder, "rcs_npy")
        
        # Skip if source folder is missing
        if not os.path.exists(rcs_txt_folder):
            continue
            
        # Create output folder
        os.makedirs(rcs_npy_folder, exist_ok=True)
        
        # Get all text files (1.txt, 2.txt, etc.)
        txt_files = glob.glob(os.path.join(rcs_txt_folder, "*.txt"))
        
        converted_count = 0
        
        for txt_file in txt_files:
            # Determine target filename
            filename = os.path.basename(txt_file)
            name_no_ext = os.path.splitext(filename)[0]
            npy_path = os.path.join(rcs_npy_folder, f"{name_no_ext}.npy")
            
            # If .npy exists, skip loop iteration
            if os.path.exists(npy_path):
                continue
            
            # Infer shape and save
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lineas = f.readlines()
                    
                if not lineas:
                    continue

                lineas = [l for l in lineas if l.strip()] 
                n_rows = len(lineas)
                
                # -------This may be useful in the future if the txt files are not square, but right now it is not necessary.----------
                
                # # 2. Calculate Columns (Angles) based on the first line
                # # Your format ends with a semicolon, so we split by ';' and remove the last empty element
                # first_line_parts = lineas[0].strip().split(';')
                # # Filter out any empty strings resulting from the trailing semicolon
                # valid_parts_sample = [p for p in first_line_parts if p.strip()]
                # n_cols = len(valid_parts_sample)
                
                # Initialize Matrix with inferred shape
                matriz = np.zeros((n_rows, n_rows, 2), dtype=np.float32)

                # Fill Matrix
                for i, linea in enumerate(lineas):
                    partes = linea.strip().split(';')
                    # Get only valid data blocks
                    valid_parts = [p for p in partes if p.strip()]
                    
                    for j, element in enumerate(valid_parts):
                        nums = element.strip().split()
                        
                        matriz[i, j, 0] = float(nums[-2]) # Using -2 and -1 is safer for whitespace
                        matriz[i, j, 1] = float(nums[-1])

                # Save the file
                # print("El npy guarda esta matriz", matriz)
                # print("La matriz tiene forma", matriz.shape)
                # print("Guardando en", npy_path)
                np.save(npy_path, matriz)
                converted_count += 1
                
            except Exception as e:
                print(f"Error converting {filename} in {os.path.basename(geo_folder)}: {e}")

        if converted_count > 0:
            print(f"  \n {os.path.basename(geo_folder)}: Generated {converted_count} new .npy files.")

    print("\nVerification complete. Npy files are up to date.")

def multiple_reorganization(nr_list, g_list):
    """
    Creates a dictionary mapping geometries to their corresponding sample numbers.
    This function allows specifying different numbers of samples for each geometry.
    If a single sample number is provided, it will be applied to all geometries.
    If multiple sample numbers are provided, each one will be mapped to its
    corresponding geometry in order.
    Parameters
    ----------
    nr_list : list
        List of sample numbers. Can be either a single-element list (same number
        for all geometries) or a list with the same length as g_list (different
        number for each geometry).
    g_list : list
        List of geometry identifiers to map sample numbers to.
    Returns
    -------
    dict
        A dictionary where keys are geometry identifiers from g_list and values
        are tuples of (0, sample_number) for each geometry.
    Raises
    ------
    AssertionError
        If nr_list length is neither 1 nor equal to g_list length.
    """
    
    assert len(nr_list) == len(g_list) or len(nr_list) == 1, "Error in multiple reorganization"
    nr_dict = {}

    if len(nr_list) == len(g_list):
        for i in range(len(g_list)):
            nr_dict[g_list[i]] = (nr_list[i])
        return nr_dict
    elif len(nr_list) == 1:
        for i in range(len(g_list)):
            nr_dict[g_list[i]] = (nr_list[0])
        return nr_dict

def generate_labeled_dataset(
    geometries,                 # List of geometry names ["Boeing...", "Rafale..."]
    samples_per_geo: dict,      # Dict: Number of samples per geometry
    case: int,                  # Source path (e.g., "data/cases/0")
    data_type,                  # "rcs_complex", "ISAR", "rcs_amp_ph", "rcs_amp", "rcs_ph"
    SNR: None | float | tuple,  # None, float (20), or tuple (10, 30)
    interval: tuple[str, float] | tuple[tuple, tuple] | None=None   #POV with cone width or 2 tuples (min, max) in degrees
                            ):
    """
    Creates a fresh, labeled dataset in 'target_path' with specific noise and data type. 
    It is designed to stored the created dataset in the target folder, 
    generating a manifest.csv file with metadata for each sample, 
    and overwriting any existing data in that folder when run again.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(script_dir, "..", "CNN", "labeled_dataset") # Target folder for the dataset (to be overwritten)

    base_path = os.path.join("data", "cases", str(case))
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    if interval:
        if type(interval[0]) == str:
            theta_lims, phi_lims = get_pov_intervals(interval[0], interval[1])
        elif type(interval[0]) == list:
            interval[1][0] = interval[1][0] % 360.0
            interval[1][1] = interval[1][1] % 360.0
            theta_lims = interval[0]
            phi_lims = interval[1]
    else:
        theta_lims = None
        phi_lims = None
    
    print("Angle limits: Theta=", theta_lims, " and Phi=", phi_lims, "\n")
    
    # Create structure: data/ folder for the actual files
    save_dir = os.path.join(target_path, "data")
    os.makedirs(save_dir)
    
    manifest_rows = []
    class_to_idx = {name: i for i, name in enumerate(geometries)}
    
    print(f"Data Type: {data_type} | SNR: {SNR} \n")
    
    for geometry in geometries:
        # Define source paths
        geo_root = os.path.join(base_path, geometry)
        csv_path = os.path.join(geo_root, "rcs.csv")
        clean_npy_path = os.path.join(geo_root, "rcs_npy")
        
        # Check if source exists
        if not os.path.exists(csv_path) or not os.path.exists(clean_npy_path):
            print(f"[WARN] Skipping {geometry}: Missing csv or rcs_npy folder.")
            continue

        df = pd.read_csv(csv_path, sep=';')
        df['phi'] = df['phi'] % 360.0 # This line must be removed, correct intervals in coordinate generation for dataset
        df['theta'] = df['theta'].clip(0, 180) # When the coordinates are generated correctly, this line will not be necessary.

        # Filter by Angles
        if theta_lims:
            df = df[(df['theta'] >= theta_lims[0]) & (df['theta'] <= theta_lims[1])]
        if phi_lims:
            p_min, p_max = phi_lims
            if p_min < p_max:
                # Standard Case (e.g., Left Side: 70 to 110), We want angles BETWEEN min and max
                df = df[(df['phi'] >= p_min) & (df['phi'] <= p_max)]
            else:
                # Wrap Case (e.g., Front: 340 to 20), We want angles GREATER than 340 OR LESS than 20
                df = df[(df['phi'] >= p_min) | (df['phi'] <= p_max)]

        # Sample Selection, prepared to receive a int or a dictionary as "samples_per_geo"
        count = samples_per_geo.get(geometry)
        print(f"{geometry}: Requested {count} samples after filtering. \n")

        if len(df) > count: 
            # If the number of available samples is greater than the requested number of samples, we randomly select the requested number of samples
            # seed = 53
            seed = datetime.now().microsecond
            df = df.sample(n=count, replace=False, random_state = seed)
        elif len(df) <= count:
            # If not, we take all the available samples
            print(f"[WARN] {geometry}: Only {len(df)} samples available after angular filter, requested {count}. Using all available samples.\n")
        elif len(df) == 0:
            print(f"[WARN] {geometry}: No samples found in angular range.\n")
            continue
            
        print(f"Processing {len(df)} samples for {geometry}...\n")

        for _, row in df.iterrows():
            file_id = str(int(row['file'])) # Remove decimals if any
            src_file = os.path.join(clean_npy_path, f"{file_id}.npy")
            
            if not os.path.exists(src_file):
                continue
                
            # A. Load Clean Data (Matrix of Real/Imag pairs)
            # Shape: (Freqs, Angles, 2)
            matriz = np.load(src_file)
            
            # Reconstruct Complex Numbers
            matriz_complex = matriz[:, :, 0] + 1j * matriz[:, :, 1]

            # print("Loaded matrix shape: ", matriz_complex.shape, matriz.shape)
            # print("Sample theta, phi: ", row['theta'], row['phi'], " | File ID: ", file_id, "from geometry ", geometry)   
            # print(matriz_complex)
            # B. Inject Noise (Random SNR if range provided, fixed otherwise)
            current_snr = SNR
            if isinstance(SNR, (tuple, list)):
                current_snr = np.random.uniform(SNR[0], SNR[1])
            
            if current_snr is not None:
                # print("SNR extracted after calculating: ", 10 * np.log10(pot_prom / pot_prom_noise), "dB. Theoretical value: ", snr, "dB.")
                potencia_promedio = np.mean(np.abs(matriz_complex)**2)
                desv = np.sqrt(potencia_promedio) / (10 ** (current_snr / 20))
                noise = np.random.normal(0, desv, matriz_complex.shape) / np.sqrt(2) + 1j * np.random.normal(0, desv, matriz_complex.shape) / np.sqrt(2)               
                matriz_complex = matriz_complex + noise

            # C. Generate Requested Format
            save_name = f"{len(manifest_rows) + 1}_{geometry}_{file_id}" # Base filename
            
            if data_type == 'rcs_complex':
                # Save as .npy (2, H, W)
                # Stack Real and Imaginary
                data_to_save = np.stack([matriz_complex.real, matriz_complex.imag], axis=0)
                final_path = os.path.join(save_dir, f"{save_name}.npy")
                np.save(final_path, data_to_save)
                
            elif data_type == 'ISAR':
                # Perform FFT
                fft_general = np.fft.fftshift(np.fft.fft2(matriz_complex))
                final_path = os.path.join(save_dir, f"{save_name}.png")
                # Save as Grayscale PNG
                plt.imsave(final_path, np.abs(fft_general), cmap="gray", format="png")
                plt.close("all")
                
            elif data_type == 'rcs_amp':
                # Amplitude only
                field_amp = np.abs(matriz_complex)
                final_path = os.path.join(save_dir, f"{save_name}.npy")
                np.save(final_path, field_amp)
                
            elif data_type == 'rcs_ph':
                # Phase only
                field_ph = np.angle(matriz_complex)
                final_path = os.path.join(save_dir, f"{save_name}.npy")
                np.save(final_path, field_ph)

            elif data_type == 'rcs_amp_ph':
                # Both Amp and Phase stacked (2, H, W)
                field_amp = np.abs(matriz_complex)
                field_ph = np.angle(matriz_complex)
                data_to_save = np.stack([field_amp, field_ph], axis=0)
                final_path = os.path.join(save_dir, f"{save_name}.npy")
                np.save(final_path, data_to_save)
            
            else:
                raise ValueError(f"Unknown data_type: {data_type}")

            if current_snr != None:
                SNR_Value = round(current_snr, 2)
            else:
                SNR_Value = "Clean"
            # D. Add to Manifest
            manifest_rows.append({
                "file_path": Path("labeled_dataset/data/") / os.path.basename(final_path),
                "label_idx": class_to_idx[geometry],
                "label_name": geometry,
                "theta": row['theta'],
                "phi": row['phi'],
                "snr": SNR_Value
            })

    # --- 4. SAVE MANIFEST ---
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv_path = os.path.join(target_path, "manifest.csv")
    manifest_df.to_csv(manifest_csv_path, index=False, sep=';')
    
    print(f"Done. Created {len(manifest_df)} samples in '{target_path}'.\n")
    print(f"Manifest saved to: {manifest_csv_path}\n")

if __name__ == '__main__':
    pass