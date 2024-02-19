import os, math, numpy as np
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier

'''
Read data function that traverses the subfolders of a provided dataset folder and reads
sample data into the program for later filtering and computations.
Parameters: none
Returned: a two dimensional array of all read-in sample data in string format | all 
            classifiers for each of the samples wihthin the dataset: 0 = male, 1 = female
'''
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

'''
Clean data function that converts read-in string data from dataset files to 
float data for later computations.
Parameters: a two dimensional array of string data read-in from each file
            within a given dataset.
Returned: a two dimensional string array of the coordinate points extracted
            from each sample file from a given dataset.
'''
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

'''
Feature extraction function that takes sample coordinate data and creates seven
features per sample. 
Parameters: a two dimensional array that contains all coordinate data points for
            all samples.
Returned: a two dimensional array of all feature data for each sample within the 
            given dataset.
'''
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

'''
Euclidean distance function that uses the euclidean distance formula to 
calculate the distance between two points on a cartesian plane.
Parameters: two arrays that represent (x,y) coordinate points in the format: [x_num,y_num]
Returned: the calculated euclidean distance between two points. 
'''
def euclidean_distance(point1, point2):
    x1 = point1[0]
    y1 = point1[1]

    x2 = point2[0]
    y2 = point2[1]

    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

'''
Data normalization function that uses the min-max normalization formula to 
convert feature data to values within the interval: [0,1].
Parameters: feature data
Returned: normalized data derived from the feature data.
'''
def normalize_data(data):
    normalized_data = []
    
    for sample in data:
        new_features = []
        sample_min = min(sample)
        sample_max = max(sample)
        for feature in sample:
            norm_feature = (feature - sample_min) / (sample_max - sample_min)
            formated_num = format(norm_feature, '.2f')
            new_features.append(float(formated_num))
        normalized_data.append(new_features)
    
    return normalized_data

'''
Creates a confusion matrix of the true male/female classes versus a model's predicted classification.
The confusion matrix is used to calculate the accuracy, precision, and recall had by a model.
Parameters: a trained classificaion model | known test-classifier data
Returned: model's accuracy, precision, and recall.
'''
def calculate_stats(model, classifiers):
    #create a dictionary to be used as a confusion matrix
    confusion_matrix = {'true_male':0, 'mf_error':0, 'fm_error':0, 'true_female':0}

    num_pred = len(model) #number of predicitons made by the model
    #populate matrix by comparing model predicitons to their known-true classes
    for i in range(num_pred):
        if(model[i] == 0 and classifiers[i] == 0):
            confusion_matrix['true_male'] += 1
        elif(model[i] == 1 and classifiers[i] == 0):
            confusion_matrix['mf_error'] += 1
        elif(model[i] == 0 and classifiers[i] == 1):
            confusion_matrix['fm_error'] += 1
        elif(model[i] == 1 and classifiers[i] == 1):
            confusion_matrix['true_female'] += 1
    
    #calculate accuracy
    numerator = confusion_matrix['true_male'] + confusion_matrix['true_female']
    denominator = num_pred
    acc = (numerator / denominator) * 100
    acc = format(acc, '.2f')

    #calculate precision
    prec_male = confusion_matrix['true_male'] / (confusion_matrix['true_male'] + confusion_matrix['fm_error'])
    prec_female = confusion_matrix['true_female'] / (confusion_matrix['true_female'] + confusion_matrix['mf_error'])
    prec = ((prec_male + prec_female) / 2 ) * 100
    prec = format(prec, '.2f')

    #calculate recall
    rec_male = confusion_matrix['true_male'] / (confusion_matrix['true_male'] + confusion_matrix['mf_error'])
    rec_female = confusion_matrix['true_female'] / (confusion_matrix['true_female'] + confusion_matrix['fm_error'])
    rec = ((rec_male + rec_female) / 2) * 100
    rec = format(rec, '.2f')

    return acc, prec, rec

'''
Main program function in which the program's logic is ran.
Parameters: none
Returned: none
'''
def main():
    #read provided data from all files within their sub-directories
    extracted_data, total_classifiers = read_data()
    #clean the data so that it can be understood as numerical data
    raw_files_data = clean_data(extracted_data)
    #use raw coordinate data to extract the features from each face sample
    total_data = extract_features(raw_files_data)
    #normalize data using min-max normalization
    normalized_data = normalize_data(total_data)

    #list to be used to divide the total samples and classifiers for training and testing
    training_data = []
    training_targets = []

    testing_data = []
    testing_targets = []

    data_size = len(normalized_data)
    num_samples = 0 #to know when the third sample of each folder is reached (max num training samples)
    for i in range(data_size):
        if(num_samples < 3): #add the sample to training data
            training_data.append(normalized_data[i])
            training_targets.append(total_classifiers[i])
            num_samples += 1
        else: #add to testing data (to test created models)
            testing_data.append(normalized_data[i])
            testing_targets.append(total_classifiers[i])
            num_samples = 0
    
    #train and test K-Nearest Neighbors Model
    num_neighbors = 18
    nn_model = neighbors.KNeighborsClassifier(num_neighbors)
    nn_model.fit(training_data, training_targets) #train model
    nn_prediction = nn_model.predict(testing_data)
    nn_acc, nn_prec, nn_rec = calculate_stats(nn_prediction, testing_targets) #calculate required statistics

    #train and test Decision Tree Model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(training_data, training_targets)
    dt_prediction = dt_model.predict(testing_data)
    dt_acc, dt_prec, dt_rec = calculate_stats(dt_prediction, testing_targets) #calculate required statistics
    
    #display results for both classification models
    print("\t", "** K-NEAREST NEIGHBORS TEST **")
    print(f'K-Neighbors: {num_neighbors}')
    print(f'Model Predicted Classes: {nn_prediction}')
    print(f'True Data Classes: {testing_targets}')
    print(f'STATISTICS - Accuracy: {nn_acc}% | Precision: {nn_prec}% | Recall {nn_rec}%', "\n")

    print("\t", "** DECISION TREE TEST **")
    print(f'Model Predicted Classes: {dt_prediction}')
    print(f'True Data Classes: {testing_targets}')
    print(f'STATISTICS - Accuracy: {dt_acc}% | Precision: {dt_prec}% | Recall {dt_rec}%')

if __name__ == "__main__":
    main()