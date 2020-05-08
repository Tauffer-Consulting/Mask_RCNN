import os
from glob import glob
from tqdm import tqdm

import itk

patients_path = glob("data/patient*/")

for patient_path in tqdm(patients_path, ncols=80):
    filenames = glob(patient_path + "*.mhd")
    for filename in filenames:
        try:
            image = itk.imread(filename)
            itk.imwrite(image, filename.replace(".mhd", ".png"))
        except:
            pass
