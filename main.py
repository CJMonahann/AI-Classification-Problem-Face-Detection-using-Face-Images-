import os
# B:\SCSU\CSC-481-Artificial-Intelligence\HW1\Face Database\m-001
# .\Face Database\m-001
def read_data():

    all_data = [] #stores all sample data from all directories within the Face Database folder
    all_data_classifiers = [] #stores the targets (classifiers) for a sample

    for dirpath, dirnames, files in os.walk('.', topdown=False):

        if(dirpath != '.' and not dirpath.startswith('.\.git')):

            print(dirpath)

            if('m' in dirpath): #if the folder has the character 'm', the classifier is 'male' as 0
                all_data_classifiers.extend([0 for i in range(4)])
            elif ('w' in dirpath): #if the folder has the character 'w', the classifier is 'woman' as 1
                all_data_classifiers.extend([1 for i in range(4)])

            for file_name in files:
                file_path = f"{dirpath}\{file_name}"
                in_file = open(file_path, 'r')
                directory_data = in_file.readlines() #reads all lines of data within a sample file and stores it in a list
                all_data.append(directory_data) #the above list of data is stored in the list containing all sample data
                in_file.close()
    
    return all_data, all_data_classifiers

def clean_data(data_list):

    all_clean_list = []

    for file_data in data_list:
        single_file = []
        fixed_file = file_data[3:]
        fixed_file.pop()

        for data in fixed_file:
            temp_arr =  []
            temp_data = data.strip("\n")
            x_coord, y_coord = temp_data.split()
            temp_arr.append(float(x_coord))
            temp_arr.append(float(y_coord))
            single_file.append(temp_arr)

        all_clean_list.append(single_file)
    
    return all_clean_list

def extract_features(feature_data):
    pass

def main():
    extracted_data, data_classifiers = read_data()
    raw_files_data = clean_data(extracted_data)
    extract_features(raw_files_data)

if __name__ == "__main__":
    main()