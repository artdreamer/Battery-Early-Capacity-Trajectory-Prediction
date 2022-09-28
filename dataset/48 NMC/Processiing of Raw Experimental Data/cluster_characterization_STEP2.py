import glob, os
import shutil

path = 'C:/Users/jinqiang/Dropbox/IEC Project Materials/Code/RWTH-2021-04545_818642/characterization/'

# create cycle and characterization folders if not exist

# if not os.path.exists(path + "Characterization after one-week cycling"):
#     os.mkdir(path + "Characterization after one-week cycling")

all_files = glob.glob(path + '*.csv')

cells = ['00' + str(i) for i in range(2, 10)] + ['0' + str(i) for i in range(10, 50)]
files_cells_BOL = {cell:[] for cell in cells}
files_cells_Characterization = {cell:[] for cell in cells}

for file in all_files:
    print(os.path.basename(file), '\n')

for file in all_files:
        terms = file.split("/")
        last_term = terms[-1]
        last_folder, file_name = last_term.split("\\")
        cell = file_name[12:15]
        if file.find("BOL Part 1") != -1:
            files_cells_BOL[cell].append(file_name)
        if file.find("=ZYK=") != -1 and file.find("=CU=") != -1:
            files_cells_Characterization[cell].append(file_name)

# for file in all_files:
#     terms = file.split("/")
#     last_term = terms[-1]
#     last_folder, file_name = last_term.split("\\")
#     cell = file_name[12:15]
#     if file.find("BOL Part 1") != -1:
#         # date = file_name[42:52]
#         if files_cells_BOL[cell][-1] == file_name:
#             os.rename(file, path + "BOL Part 1/" + file_name)
#
    # if file.find("=ZYK=") != -1:
    #     # date = file_name[35:45]
    #     if files_cells_Characterization[cell][0] == file_name:
    #         os.rename(file, path + "Characterization after one-week cycling/" + file_name)

week = 0
for cell in cells:
    if not os.path.exists(path + "BOL Part 1"):
        os.mkdir(path + "BOL Part 1")
    filename = files_cells_BOL[cell][-1]
    os.rename(path + filename, path + "BOL Part 1/" + filename)

week = 1
for week in range(1, 30):
    for cell in cells:
        if len(files_cells_Characterization[cell]) >= week:
            if not os.path.exists(path + f"Characterization after {str(week)} weeks (6.5 days) of cycling"):
                os.mkdir(path + f"Characterization after {str(week)} weeks (6.5 days) of cycling")
            filename = files_cells_Characterization[cell][week-1]
            os.rename(path + filename, path + f"Characterization after {str(week)} weeks (6.5 days) of cycling/" + filename)