import glob, os

path = 'C:/Users/jinqiang/Dropbox/IEC Project Materials/Code/RWTH-2021-04545_818642/'

# create cycle and characterization folders if not exist
if not os.path.exists(path + "cycling"):
    os.mkdir(path + "cycling")
if not os.path.exists(path + "characterization"):
    os.mkdir(path + "characterization")

all_files = glob.glob(path + '*.csv')
cycle_test_files, characterization_test_files = [], []
for file in all_files:
    if file.find("Zyk") != -1:
        terms = file.split("/")
        last_term = terms[-1]
        last_folder, file_name = last_term.split("\\")
        cycle_test_files.append(path + 'cycling/' + file_name)
        os.rename(file, path + 'cycling/' + file_name)
    else:
        terms = file.split("/")
        last_term = terms[-1]
        last_folder, file_name = last_term.split("\\")
        characterization_test_files.append(path + 'characterization/' + file_name)
        os.rename(file, path + 'characterization/' + file_name)
print(len(cycle_test_files))
print(file_name)
