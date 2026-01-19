import argparse
from data.generator import fill_stl, change_num_facets, add_stl, remove_stl, create_case, calculate_rcs, generate_labeled_dataset, multiple_reorganization, npy_fill
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments to send them to the various necessary programmes.')
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # Creación de subparsers
    parser_add = subparsers.add_parser("addstl", help="Adds a new STL to the CSV file that has been placed in data/stl.")
    parser_rm = subparsers.add_parser("rmstl", help="Removes a certain STL from the CSV")
    parser_fill = subparsers.add_parser("fillstl", help="Fills in the auxiliary data in stl.csv")
    parser_nfacets = subparsers.add_parser("nfacets", help="Changes the number of facets in a certain STL. Doesn't change the CSV.")
    parser_case = subparsers.add_parser("case", help="Creates a new case in param.csv for ISAR image generation (With the corresponding and necessary folders).")
    parser_rcs = subparsers.add_parser("rcs", help="Creates new samples for a specific case")
    parser_label = subparsers.add_parser("label", help="Create a dataset with its corresponding labels")
    
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
    parser_rcs.add_argument('-g', '--geometries', nargs='+', type=str, help='List of .stl files')
    parser_rcs.add_argument('-c', '--case', type=int, help='Case number to create or use.')
    parser_rcs.add_argument('-n', '--num_samples', type=int, help='Number of samples for each stl.')
    # First couple
    parser_rcs.add_argument('--pov', type=str, default=None, choices=["front", "left_side", "right_side", "back", "top", "bottom"], help='Which part of the object will be being looked')
    parser_rcs.add_argument('--cw', type=float, default=None, help='Value of the semi-width established for the pov cone in degrees')
    # Second couple
    parser_rcs.add_argument('--theta', nargs=2, type=float, default=None, help='Values of theta interval')
    parser_rcs.add_argument('--phi', nargs=2, type=float, default=None, help='Values of phi interval')

    # SUBPARSER LABEL
    parser_label.add_argument('-g', '--geometries', nargs='+', type=str, help='List of .stl files')
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
        # Initial check: either pov and cw have been given, or theta and phi, or neither.
        if not any([args.pov, args.cw, args.theta, args.phi]):
            interval = None
        # pov y cw have been declared
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

# nohup python generate.py rcs -g  -n  -c  --pov  --cw 
# python generate.py label -g  -n  -c  --data  --SNR  --pov  --cw
