import subprocess
import logging
import os
from datetime import datetime

actual_time = datetime.now()

userPath = os.getcwd().split('/')[2]
if userPath == "newfasant2":
    userPath = userPath + "/N101"

logs_folder_path=f'/home/{userPath}/N101-IA/CNN/SuperLogs'
os.makedirs(logs_folder_path, exist_ok=True)
logging.basicConfig(
    filename=f'/home/{userPath}/N101-IA/CNN/SuperLogs/Day_{actual_time.day}_{actual_time.month}_{actual_time.year}_Time_{actual_time.hour:02d}_{actual_time.minute:02d}_dataset.log',  # Name of the log file
    level=logging.INFO,  # Logs level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

"""
Lista de posibles inputs:
- npy: La RCS como tensor m x n x 2 (m=angulos ,n=frecs, 2=complejos)
- field: lo mismo que npy, pero con amplitud y fase
- ISAR: imagen ISAR en PNG
- field_ph_npy: lo mismo que field pero solo fase
"""
list_types = ["npy", "field", "ISAR", "field_ph_npy"]
list_epochs = ["61", "65", "71", "150"]

for i in range(len(list_types)):
    for j in range(len(list_epochs)):
        result = subprocess.run(["python", "/home/newfasant2/N101/N101-IA/CNN/CNN_general.py", "-i","/home/newfasant2/N101/N101-IA/Datasets/Reorganized/Classification_4500_0_16_f_16_d_POV_90.0_SNR_10.0",
                         "-d", list_types[i], "-e", list_epochs[j]], capture_output=True, text=True)
        print(result.stdout,"\n")
        logging.info(result.stdout)