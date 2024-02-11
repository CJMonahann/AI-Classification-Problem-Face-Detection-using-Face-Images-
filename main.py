import os, math
# B:\SCSU\CSC-481-Artificial-Intelligence\HW1\Face Database\m-001
# .\Face Database\m-001
def read_data():

    all_data = [] #stores all sample data from all directories within the Face Database folder
    all_data_classifiers = [] #stores the targets (classifiers) for a sample

    for dirpath, dirnames, files in os.walk('.', topdown=False):

        if(dirpath != '.' and not dirpath.startswith('.\.git')):

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

def extract_features(coord_data):

    all_features = []

    for file_coords in coord_data:
        temp_features = [] # a list to hold a single file's feature data

        #1.) calculate eye length ratio feature

        #determine wich eye is longer (use max length for calculation)
        eye1 = euclidean_distance(file_coords[9],file_coords[10])
        eye2 = euclidean_distance(file_coords[11],file_coords[12])

        if(eye1 > eye2):
            numerator = eye1
        else:
            numerator = eye2

        denominator = euclidean_distance(file_coords[8], file_coords[13])
        calculation = numerator / denominator
        feature = format(calculation, '.2f')
        temp_features.append(float(feature))

        #2.) calculate eye distance ratio feature
        numerator = euclidean_distance(file_coords[0], file_coords[1])
        denominator = euclidean_distance(file_coords[8], file_coords[13])
        calculation = numerator / denominator
        feature = format(calculation, '.2f')
        temp_features.append(float(feature))

        #3.) calculate nose ratio feature
        numerator = euclidean_distance(file_coords[15], file_coords[16])
        denominator = euclidean_distance(file_coords[20], file_coords[21])
        calculation = numerator / denominator
        feature = format(calculation, '.2f')
        temp_features.append(float(feature))

        #4.) calculate lip size ratio feature
        numerator = euclidean_distance(file_coords[2], file_coords[3])
        denominator = euclidean_distance(file_coords[17], file_coords[18])
        calculation = numerator / denominator
        feature = format(calculation, '.2f')
        temp_features.append(float(feature))

        #5.) calculate lip length ratio feature
        numerator = euclidean_distance(file_coords[2], file_coords[3])
        denominator = euclidean_distance(file_coords[20], file_coords[21])
        calculation = numerator / denominator
        feature = format(calculation, '.2f')
        temp_features.append(float(feature))

        #6.) calculate eye-brow length ratio feature
        brow1 = euclidean_distance(file_coords[4],file_coords[5])
        brow2 = euclidean_distance(file_coords[6],file_coords[7])

        if(brow1 > brow2):
            numerator = brow1
        else:
            numerator = brow2

        denominator = euclidean_distance(file_coords[8], file_coords[13])
        calculation = numerator / denominator
        feature = format(calculation, '.2f')
        temp_features.append(float(feature))
        
        #7.) calculate aggressive ratio feature
        numerator = euclidean_distance(file_coords[10], file_coords[19])
        denominator = euclidean_distance(file_coords[20], file_coords[21])
        calculation = numerator / denominator
        feature = format(calculation, '.2f')
        temp_features.append(float(feature))

        all_features.append(temp_features)

    return all_features

def euclidean_distance(point1, point2):
    x1 = point1[0]
    y1 = point1[1]

    x2 = point2[0]
    y2 = point2[1]

    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def main():
    #read provided data from all files within their sub-directories
    extracted_data, total_classifiers = read_data()
    #clean the data so that it can be understood as numerical data
    raw_files_data = clean_data(extracted_data)
    #use raw coordinate data to extract the features from each face sample
    total_data = extract_features(raw_files_data)

if __name__ == "__main__":
    main()