import os
# B:\SCSU\CSC-481-Artificial-Intelligence\HW1\Face Database\m-001
# .\Face Database\m-001
def read_data():

    all_data = [] #stores all sample data from all directories within the Face Database folder
    all_data_classifiers = [] #stores the targets (classifiers) for a sample

    for dirpath, dirnames, files in os.walk('.', topdown=False):


        #print(f'Found directory: {dirpath}')

        if('m' in dirpath): #if the folder has the character 'm', the classifier is 'male', 'm'.
            all_data_classifiers.append('m')
        else: #if the folder has the character 'w', the classifier is 'woman', 'w'
            all_data_classifiers.append('w')
            

        if(dirpath != '.'):
            for file_name in files:
                #print(file_name)
                file_path = f"{dirpath}\{file_name}"
                in_file = open(file_path, 'r')
                directory_data = in_file.readlines() #reads all lines of data within a sample file and stores it in a list
                all_data.append(directory_data) #the above list of data is stored in the list containing all sample data
                in_file.close()
    
    return all_data, all_data_classifiers

def clean_data(data_list):
    for file_data in data_list:
        for file_element in file_data:
            
            print(file_element, "_______________________________")

def main():
    extracted_data, data_classifiers = read_data()
    clean_data(extracted_data)


if __name__ == "__main__":
    main()