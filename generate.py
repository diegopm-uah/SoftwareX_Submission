import argparse
import logging
import os
from datetime import datetime
from data.generator import fill_stl, change_num_facets, add_stl, remove_stl, create_case, calculate_rcs, generate_labeled_dataset, multiple_reorganization, npy_fill
from pathlib import Path

# Getting day and time
actual_time = datetime.now()
logs_folder_path=Path(f'data/logs')
os.makedirs(logs_folder_path, exist_ok=True)
logging.basicConfig(
    filename=logs_folder_path / f"{actual_time.day}_{actual_time.month}_{actual_time.year}_Time_{actual_time.hour:02d}_{actual_time.minute:02d}_dataset.log",  # Name of the log file
    level=logging.INFO,  # Logs level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

tic_g = datetime.now()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Procesar argumentos para enviarlos a los distintos programas necesarios.')

    # We differentiate by 'mode' (stl, sim, label)
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # ==========================================
    # ROAD 1: STL MANAGER
    # ==========================================
    parser_stl = subparsers.add_parser("stl", help="Manage Geometry Files (Add/Remove/Facets)")
    stl_subparsers = parser_stl.add_subparsers(dest='stl_command', required=True)

    # 1.1 Add
    parser_add = stl_subparsers.add_parser("add", help="Add new STL to CSV")
    parser_add.add_argument('geometries', nargs='+', type=str, help='Name of the .STL geometries to be added')
    parser_add.add_argument('-c', '--category', type=str, default='', help='Category to which the new STL belongs')

    # 1.2 Remove
    parser_rm = stl_subparsers.add_parser("rm", help="Remove STL from CSV")
    parser_rm.add_argument('geometries', nargs='+', type=str, help='Name of the .STL geometries to be removed')

    # 1.3 Fill
    parser_fill = stl_subparsers.add_parser("fill", help="Fill stl CSV data")

    # 1.4 NFacets
    parser_nfacets = stl_subparsers.add_parser("nfacets", help="Change facet count")
    parser_nfacets.add_argument('geometries', nargs='+', type=str, help='Name of the geometries STL')
    parser_nfacets.add_argument('-n', '--num_facets', type=int, help='Target facets number for the STL')

    # ==========================================
    # ROAD 2: SIMULATIONS MANAGER
    # ==========================================

    parser_sim = subparsers.add_parser("sim", help="Simulation Configuration and Execution")
    sim_subparsers = parser_sim.add_subparsers(dest='sim_command', required=True)

    # 2.1 CASE (Parameter configuration)
    parser_case = sim_subparsers.add_parser("case", help="Create simulation case parameters")
    parser_case.add_argument("sweep", choices=['theta', 'phi'], help='Sweep direction')
    parser_case.add_argument('-da', '--delta_angle', required=True, type=float, help='Angular sweep interval width')
    parser_case.add_argument('-df', '--delta_freq', required=True, type=float, help='Frequency sweep interval width')
    parser_case.add_argument('-cf', '--central_freq', required=True, type=float, help='Central frequency')
    parser_case.add_argument('-na', '--num_angle', required=True, type=int, help="Number of angles per angular sweep")
    parser_case.add_argument('-nf', '--num_freq', type=int, required=True, help="Number of frequencies per frequency sweep")

    # 2.2 RCS (RCS Generation)
    parser_rcs = sim_subparsers.add_parser("rcs", help="Run RCS calculation")
    parser_rcs.add_argument('-g', '--geometries', nargs='+', type=str, help='Lista de archivos .msh y .stl')
    parser_rcs.add_argument('-c', '--case', type=int, help='Case number to create or use.')
    parser_rcs.add_argument('-n', '--num_samples', type=int, help='Number of samples for each stl.')
    # First couple
    parser_rcs.add_argument('--pov', type=str, default=None, choices=["front", "left_side", "right_side", "back", "top", "bottom"], help='Which part of the object will be being looked')
    parser_rcs.add_argument('--cw', type=float, default=None, help='Value of the semi-width established for the pov cone in degrees')
    # Second couple
    parser_rcs.add_argument('--theta', nargs=2, type=float, default=None, help='Values of theta interval')
    parser_rcs.add_argument('--phi', nargs=2, type=float, default=None, help='Values of phi interval')

    # ==========================================
    # ROAD 3: LABEL GENERATION (label)
    # ==========================================

    parser_label = subparsers.add_parser("label", help="Generate Labeled Dataset")

    parser_label.add_argument('-g', '--geometries', nargs='+', type=str, help='Lista de archivos .stl')
    parser_label.add_argument('-n', '--num_samples', nargs='+', type= int, help='Number of samples for each stl (1 "int" = same number for all geometries, + "int" = different number for each geometry).')
    parser_label.add_argument('-c', '--case', type=int, help='Case number to get data from.')
    parser_label.add_argument('--data', type=str, required=True,choices=['rcs_complex', 'ISAR', 'rcs_amp_ph', 'rcs_amp', 'rcs_ph'], help='Type of data to be generated.')
    parser_label.add_argument('--SNR', type=float, default=None, help='Value of the signal-to-noise ratio (expressed in dB) used to add noise to the dataset.')
    # First couple
    parser_label.add_argument('--pov', type=str, default=None, choices=["front", "left_side", "right_side", "back", "top", "bottom"], help='Which part of the object will the data belong to')
    parser_label.add_argument('--cw', type=float, default=None, help='Value of the semi-width established for the pov cone in degrees')
    # Second couple
    parser_label.add_argument('--theta', nargs=2, type=float, default=None, help='Values of theta interval')
    parser_label.add_argument('--phi', nargs=2, type=float, default=None, help='Values of phi interval')

    args = parser.parse_args()

    print(f'\nArguments for generate.py are:\n {vars(args)} \n')
    logging.info(f'Arguments for generate.py are: {vars(args)}')

    # Logic Road 1
    if args.mode == 'stl':
        if args.stl_command == 'add':
            for geometry in args.geometries:
                add_stl(geometry, args.category)
        elif args.stl_command == 'rm':
            for geometry in args.geometries:
                remove_stl(geometry)
        elif args.stl_command == 'fill':
            fill_stl()
        elif args.stl_command == 'nfacets':
            change_num_facets(args.geometries, args.num_facets)
    
    # Logic Road 2
    elif args.mode == 'sim':
        if args.sim_command == 'case':
            # Create case with given parameters
            create_case(args.sweep,
                        delta_angle=args.delta_angle, delta_freq=args.delta_freq,
                        central_freq=args.central_freq,
                        num_angle=args.num_angle, num_freq=args.num_freq)
        elif args.sim_command == 'rcs':
            # Initial check: either pov and cw have been given, or theta and phi, or neither.
            if not any([args.pov, args.cw, args.theta, args.phi]):
                interval = None
            
            elif args.pov and args.cw:          # pov and cw have been declared.
                if args.theta or args.phi:
                    logging.error("Theta and Phi cannot be declared simultanously with POV and CW")
                    raise Exception("Theta and Phi cannot be declared simultanously with POV and CW")
                interval = (args.pov, args.cw)
            elif args.theta and args.phi:       # theta and phi have been declared.
                if args.pov or args.cw:
                    logging.error("Theta and Phi cannot be declared simultanously with POV and CW")
                    raise Exception("Theta and Phi cannot be declared simultanously with POV and CW")
                interval = (args.theta, args.phi)
            else:                               # Error: only one of the two parameters has been declared.
                logging.error("Theta/Phi or POV/CW must be declared simultanously")
                raise Exception("Theta/Phi or POV/CW must be declared simultanously")
            
            # Run RCS calculation with given parameters and interval
            calculate_rcs(geometries=args.geometries, num_samples=args.num_samples, 
                            case=args.case, interval=interval)
            npy_fill(case=args.case)
    
    # Logic Road 3
    elif args.mode == 'label':
        # Initial check: either pov and cw have been given, or theta and phi, or neither.
        if not any([args.pov, args.cw, args.theta, args.phi]):
            interval = None

        elif args.pov and args.cw:        # pov and cw have been declared.
            if args.theta or args.phi:
                logging.error("Theta and Phi cannot be declared simultanously with POV and CW")
                raise Exception("Theta and Phi cannot be declared simultanously with POV and CW")
            interval = (args.pov, args.cw)
        elif args.theta and args.phi:     # theta and phi have been declared.
            if args.pov or args.cw:
                logging.error("Theta and Phi cannot be declared simultanously with POV and CW")
                raise Exception("Theta and Phi cannot be declared simultanously with POV and CW")
            interval = (args.theta, args.phi)
        else:                             # Error: only one of the two parameters has been declared.
            logging.error("Theta/Phi or POV/CW must be declared simultanously")
            raise Exception("Theta/Phi or POV/CW must be declared simultanously")
        
        # Generate labeled dataset with given parameters and interval
        nr_dictionary = multiple_reorganization(args.num_samples, args.geometries)
        generate_labeled_dataset(geometries = args.geometries, 
                                    samples_per_geo = nr_dictionary, 
                                    case = args.case,
                                    data_type = args.data, 
                                    SNR = args.SNR,
                                    interval = interval
                                )

    toc_g = datetime.now()
    duration = toc_g - tic_g
    total_seconds = int(duration.total_seconds())

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Create a clean string. Format example: "02h 15m 10s"
    time_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    print(f"Execution completed successfully in {time_str}")
    logging.info(f"Execution completed successfully in {time_str}")
    logging.shutdown()

# --------------- 10000 facets Figthers ----------------

# Boeing_EA-18G_Growler_10000 Dassault_Rafale_10000 Eurofighter_Typhoon_20000 F-4_Phantom_II_10000
# Grumman_EA-6B_Prowler_10000 Grumman_F-14_Tomcat_20000 JF-17_Thunder_10000 KAI_T-50_Golden_Eagle_10000 
# Lockheed_F-16_Fighting_Falcon_10000 Lockheed_F-22_Raptor_10000 Mikoyan_MiG-23_10000 Mikoyan_MiG-31_10000 
# Mikoyan_MiG-35_10000 Northrop_F-5_10000 Northrop_F-20_Tigershark_10000 Northrop_T-38_Talon_10000 
# Rockwell_B-1_Lancer_20000 Shenyang_J-15_10000 Sukhoi_SU-25_10000 Sukhoi_SU-34_10000 Sukhoi_SU-57_10000 
# TAI_Hurjet_10000 TAI_Kaan_10000 Yakovlev_Yak-130_10000

# One line format:
# Boeing_EA-18G_Growler_10000 Dassault_Rafale_10000 Eurofighter_Typhoon_20000 F-4_Phantom_II_10000 Grumman_EA-6B_Prowler_10000 Grumman_F-14_Tomcat_20000 JF-17_Thunder_10000 KAI_T-50_Golden_Eagle_10000 Lockheed_F-16_Fighting_Falcon_10000 Lockheed_F-22_Raptor_10000 Mikoyan_MiG-23_10000 Mikoyan_MiG-31_10000 Mikoyan_MiG-35_10000 Northrop_F-5_10000 Northrop_F-20_Tigershark_10000 Northrop_T-38_Talon_10000 Rockwell_B-1_Lancer_20000 Shenyang_J-15_10000 Sukhoi_SU-25_10000 Sukhoi_SU-34_10000 Sukhoi_SU-57_10000 TAI_Hurjet_10000 TAI_Kaan_10000 Yakovlev_Yak-130_10000

# --------------- 20000 facets Small Geometries  ----------------

# Vapor_55_UAV_20000 Scaneagle_UAV_20000 Drone_X8_Octocopter_20000 Agriculteur_UAV_20000 
# Glider_20000 Mehmet_Prototype_UAV_20000 IAI_Harop_20000 Helicopter_Drone_UAV_20000

# One line format:
# Vapor_55_UAV_20000 Scaneagle_UAV_20000 Drone_X8_Octocopter_20000 Agriculteur_UAV_20000 Glider_20000 Mehmet_Prototype_UAV_20000 IAI_Harop_20000 Helicopter_Drone_UAV_20000

# ----------------------------- 10000 facets Big Geometries -----------------------------

# Antonov_An-72_10000 Antonov_An-225_10000 Supersonic_Jet_10000 Sphinx_UAV_10000 Northrop_YB-35_10000 
# Northrop_Grumman_RQ-4_Global_Hawk_10000 Lockheed_U-2_10000 Lockheed_P-3_Orion_10000 Kamov_Ka-52_10000 
# HAL_Prachand_10000 Grumman_F7F_Tigercat_10000 Fairchild_A-10_Warthog_10000 Embraer_Phenom_100_10000 
# Embraer_C-390_Millennium_10000 Consolidated_PBY_Catalina_10000 Concorde_10000 Boeing_787_10000 
# Bell_V-280_Valor_10000 Bell-Boeing_V-22_Osprey_10000 Bell_AH-1Z_Viper_10000 Bayraktar_TB2_10000 
# Bayraktar_Akinci_10000 Avenger-716_UAV_10000

# One line format:
# Antonov_An-72_10000 Antonov_An-225_10000 Supersonic_Jet_20000 Sphinx_UAV_10000 Northrop_YB-35_10000 Northrop_Grumman_RQ-4_Global_Hawk_10000 Lockheed_U-2_10000 Lockheed_P-3_Orion_10000 Kamov_Ka-52_10000 HAL_Prachand_20000 Grumman_F7F_Tigercat_10000 Fairchild_A-10_Warthog_10000 Embraer_Phenom_100_20000 Embraer_C-390_Millennium_10000 Consolidated_PBY_Catalina_10000 Boeing_787_10000 Bell_V-280_Valor_10000 Bell-Boeing_V-22_Osprey_10000 Bell_AH-1Z_Viper_20000 Bayraktar_TB2_10000 Bayraktar_Akinci_10000 Avenger-716_UAV_10000

# python generate.py label -g  -n  -c  --data  --SNR  --pov  --cw
# nohup python generate.py sim rcs -g  -n  -c  --pov  --cw 

# python generate.py label -g Vapor_55_UAV_20000 Scaneagle_UAV_20000 Drone_X8_Octocopter_20000 Agriculteur_UAV_20000 Glider_20000 Mehmet_Prototype_UAV_20000 IAI_Harop_20000 Helicopter_Drone_UAV_20000 -n 2000 -c 1 --data rcs_amp_phase --pov back --cw 56 --SNR 0
