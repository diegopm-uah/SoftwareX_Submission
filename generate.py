import argparse
from data.generator import fill_stl, change_num_facets, add_stl, remove_stl, create_case, calculate_rcs, generate_labeled_dataset, multiple_reorganization, npy_fill
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Procesar argumentos para enviarlos a los distintos programas necesarios.')
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # Creación de subparsers
    parser_add = subparsers.add_parser("addstl", help="Añade un nuevo STL al CSV que se ha metido en data/stl")
    parser_rm = subparsers.add_parser("rmstl", help="Elimina un cierto STL del CSV")
    parser_fill = subparsers.add_parser("fillstl", help="Rellena los datos auxiliares en stl.csv")
    parser_nfacets = subparsers.add_parser("nfacets", help="Cambia el número de facetas en un cierto STL. No cambia el CSV.")
    parser_case = subparsers.add_parser("case", help="Crea un nuevo caso en param.csv para la generación de imagenes ISAR")
    parser_rcs = subparsers.add_parser("rcs", help="Crea nuevos samples para un cierto caso")
    parser_label = subparsers.add_parser("label", help="Crea un dataset con sus correspondientes etiquetas")
    
    # SUBPARSER ADD_STL
    parser_add.add_argument('geometries', nargs='+', type=str, help='Name of the geometries STL')
    parser_add.add_argument('-c', '--category', type=str, required=False, default='', help='Category to which the new STL belongs')
    
    # SUBPARSER REMOVE_STL
    parser_rm.add_argument('geometries', nargs='+', type=str, help='Name of the geometries STL')

    # SUBPARSER FILL: no recibe parámetros extra
    
    # SUBPARSER NFACETS
    parser_nfacets.add_argument('geometries', nargs='+', type=str, help='Name of the geometries STL')
    parser_nfacets.add_argument('-n', '--num_facets', type=int, help='Number of facets of the STL')
    
    # SUBPARSER CASE
    parser_case.add_argument("sweep", choices=['theta', 'phi'], help='Sweep direction')
    parser_case.add_argument('-da', '--delta_angle', required=True, type=float, help='Angle width')
    parser_case.add_argument('-df', '--delta_freq', required=True, type=float, help='Frequency width')
    parser_case.add_argument('-cf', '--central_freq', required=True, type=float, help='Central frequency')
    parser_case.add_argument('-na', '--num_angle', required=True, type=int, help="Number of angles per image")
    parser_case.add_argument('-nf', '--num_freq', type=int, required=True, help="Number of frequencies per image")
    
    # SUBPARSER RCS
    parser_rcs.add_argument('-g', '--geometries', nargs='+', type=str, help='Lista de archivos .msh y .stl')
    parser_rcs.add_argument('-c', '--case', type=int, help='Case number to create or use.')
    parser_rcs.add_argument('-n', '--num_samples', type=int, help='Number of samples for each stl.')
    # First couple
    parser_rcs.add_argument('--pov', type=str, default=None, choices=["front", "left_side", "right_side", "back", "top", "bottom"], help='Which part of the object will be being looked')
    parser_rcs.add_argument('--cw', type=float, default=None, help='Value of the semi-width established for the pov cone in degrees')
    # Second couple
    parser_rcs.add_argument('--theta', nargs=2, type=float, default=None, help='Values of theta interval')
    parser_rcs.add_argument('--phi', nargs=2, type=float, default=None, help='Values of phi interval')

    # SUBPARSER LABEL
    parser_label.add_argument('-g', '--geometries', nargs='+', type=str, help='Lista de archivos .stl')
    parser_label.add_argument('-n', '--num_samples', nargs='+', type= int, help='Number of samples for each stl (1 "int" = same number for all geometries, + "int" = different number for each geometry).')
    parser_label.add_argument('-c', '--case', type=int, help='Case number to get data from.')
    parser_label.add_argument('--data', type=str, required=True,choices=['rcs_complex', 'ISAR', 'rcs_amp_ph', 'rcs_amp', 'rcs_ph'], help='Type of data to be generated.')
    parser_label.add_argument('--SNR', type=float, default=None, help='Value of the signal-to-noise ratio, expressed in dB, used to add noise to the dataset.')
    # First couple
    parser_label.add_argument('--pov', type=str, default=None, choices=["front", "left_side", "right_side", "back", "top", "bottom"], help='Which part of the object will be being looked')
    parser_label.add_argument('--cw', type=float, default=None, help='Value of the semi-width established for the pov cone in degrees')
    # Second couple
    parser_label.add_argument('--theta', nargs=2, type=float, default=None, help='Values of theta interval')
    parser_label.add_argument('--phi', nargs=2, type=float, default=None, help='Values of phi interval')

    args = parser.parse_args()

    print(f'\nArguments for generate.py are:\n {vars(args)} \n')
    
    if args.mode == 'addstl':
        for geometry in args.geometries:
            add_stl(geometry, args.category)
    elif args.mode == 'rmstl':
        for geometry in args.geometries:
            remove_stl(geometry)
    elif args.mode == 'fillstl':
        fill_stl()
    elif args.mode == 'nfacets':
        change_num_facets(args.geometries, args.num_facets)
    elif args.mode == 'case':
        create_case(args.sweep,
                    delta_angle=args.delta_angle, delta_freq=args.delta_freq,
                    central_freq=args.central_freq,
                    num_angle=args.num_angle, num_freq=args.num_freq)
    elif args.mode == 'rcs' or args.mode == 'label':
        # Comprobación incial: o bien se han dado pov y cw, o bien theta y phi, o bien ninguno
        if not any([args.pov, args.cw, args.theta, args.phi]):
            interval = None
        # Se han declarado pov y cw
        elif args.pov and args.cw:
            if args.theta or args.phi:
                raise Exception("Theta and Phi cannot be declared simultanously with POV and CW")
            interval = (args.pov, args.cw)
        elif args.theta and args.phi:
            if args.pov or args.cw:
                raise Exception("Theta and Phi cannot be declared simultanously with POV and CW")
            interval = (args.theta, args.phi)
        else:
            raise Exception("Theta/Phi or POV/CW must be declared simultanously")
        if args.mode == 'rcs':
            calculate_rcs(geometries = args.geometries, 
                          num_samples = args.num_samples, 
                          case = args.case,
                          interval = interval
                         )
            npy_fill(case = args.case)
        elif args.mode =='label':
            nr_dictionary = multiple_reorganization(args.num_samples, args.geometries)
            generate_labeled_dataset(geometries = args.geometries, 
                                     samples_per_geo = nr_dictionary, 
                                     case = args.case,
                                     data_type = args.data, 
                                     SNR = args.SNR,
                                     interval = interval
                                    )

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
# nohup python generate.py calculate_rcs -g  -n  -c  --pov  --cw 

# python generate.py label -g Vapor_55_UAV_20000 Scaneagle_UAV_20000 Drone_X8_Octocopter_20000 Agriculteur_UAV_20000 Glider_20000 Mehmet_Prototype_UAV_20000 IAI_Harop_20000 Helicopter_Drone_UAV_20000 -n 2000 -c 1 --data rcs_amp_phase --pov back --cw 56 --SNR 0
