
import matplotlib.pyplot as plt
from loop_folder import get_train_errors, get_test_errors, get_min_train_errors, get_min_test_errors

'''
to get the data as a list, we can call the fucntions below:
get_train_errors(filenames_meta, files_nb)
get_test_errors(filenames_meta, files_nb)
get_min_train_errors(filenames_other, files_nb)
get_min_test_errors(filenames_other, files_nb)

filenames_meta/filenames_other is the address of the files:
example:
filenames_meta = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp2_tower_mdl/mdl_random_mdl_{}/meta_data.yml'
filenames_other = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp2_tower_mdl/mdl_random_mdl_{}/other_data.yml'

get_train_errors() and get_test_errors() need the file ending with meta_data.yml.
get_min_train_errors() and get_min_test_errors() need the file ending with other_data.yml.

files_nb: the number of files for loop, usual the number in the folder name.

'''


def get_difference(test_errors,train_errors):
    '''
    The fucntion to get the diffrences of test_errors and train_errors.
    '''
    differnce_list = []
    for i in range(len(test_errors)):
        differnce_list.append(test_errors[i] - train_errors[i])
    return differnce_list


'''
where you get the data that you want to plot, ususal the list
Take advatange of fucntions in the loop_folder to:
example:
    train_errors = get_train_errors(filenames_meta,files_nb)
    test_errors = get_test_errors(filenames_meta, files_nb)
    min_train_errors = get_min_train_errors(filenames_other, files_nb)
    min_test_errors = get_min_test_errors(filenames_other, files_nb)
    differences = get_difference(test_errors,train_errors)
    differences_of_min = get_difference(min_test_errors,min_train_errors)
'''

#group10 and Grp9 are conv tower
filenames_meta_10 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp10_conv_tower_mdl/mdl_conv_tower_mdl_{}/meta_data.yml'
filenames_other_10 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp10_conv_tower_mdl/mdl_conv_tower_mdl_{}/other_data.yml'
files_nb_10 = 233


train_errors_10 = get_train_errors(filenames_meta_10,files_nb_10)
test_errors_10 = get_test_errors(filenames_meta_10, files_nb_10)
min_train_errors_10 = get_min_train_errors(filenames_other_10, files_nb_10)
min_test_errors_10 = get_min_test_errors(filenames_other_10, files_nb_10)
differences_10 = get_difference(test_errors_10,train_errors_10)
differences_of_min_10 = get_difference(min_test_errors_10,min_train_errors_10)

filenames_meta_9 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp9_conv_tower_mdl/mdl_conv_tower_mdl_{}/meta_data.yml'
filenames_other_9 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp9_conv_tower_mdl/mdl_conv_tower_mdl_{}/other_data.yml'
files_nb_9 = 144

train_errors_9 = get_train_errors(filenames_meta_9,files_nb_9)
test_errors_9 = get_test_errors(filenames_meta_9, files_nb_9)
min_train_errors_9 = get_min_train_errors(filenames_other_9, files_nb_9)
min_test_errors_9 = get_min_test_errors(filenames_other_9, files_nb_9)
differences_9 = get_difference(test_errors_9,train_errors_9)
differences_of_min_9 = get_difference(min_test_errors_9,min_train_errors_9)


#Grp3 and Grp12, Grp2 are random:
filenames_meta_3 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp3_ramdon_mdl/mdl_random_mdl_{}/meta_data.yml'
filenames_other_3 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp3_ramdon_mdl/mdl_random_mdl_{}/other_data.yml'
files_nb_3 = 103

train_errors_3 = get_train_errors(filenames_meta_3,files_nb_3)
test_errors_3 = get_test_errors(filenames_meta_3, files_nb_3)
min_train_errors_3 = get_min_train_errors(filenames_other_3, files_nb_3)
min_test_errors_3 = get_min_test_errors(filenames_other_3, files_nb_3)
differences_3 = get_difference(test_errors_3,train_errors_3)
differences_of_min_3 = get_difference(min_test_errors_3,min_train_errors_3)

filenames_meta_12 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp12_ramdon_mdl/mdl_random_mdl_{}/meta_data.yml'
filenames_other_12 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp12_ramdon_mdl/mdl_random_mdl_{}/other_data.yml'
files_nb_12 = 107

train_errors_12 = get_train_errors(filenames_meta_12,files_nb_12)
test_errors_12 = get_test_errors(filenames_meta_12, files_nb_12)
min_train_errors_12 = get_min_train_errors(filenames_other_12, files_nb_12)
min_test_errors_12 = get_min_test_errors(filenames_other_12, files_nb_12)
differences_12 = get_difference(test_errors_12,train_errors_12)
differences_of_min_12 = get_difference(min_test_errors_12,min_train_errors_12)

filenames_meta_2 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp2_ramdon_mdl_Brando2/mdl_random_mdl_{}/meta_data.yml'
filenames_other_2 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp2_ramdon_mdl_Brando2/mdl_random_mdl_{}/other_data.yml'
files_nb_2 = 382

train_errors_2 = get_train_errors(filenames_meta_2,files_nb_2)
test_errors_2 = get_test_errors(filenames_meta_2, files_nb_2)
min_train_errors_2 = get_min_train_errors(filenames_other_2, files_nb_2)
min_test_errors_2 = get_min_test_errors(filenames_other_2, files_nb_2)
differences_2 = get_difference(test_errors_2,train_errors_2)
differences_of_min_2 = get_difference(min_test_errors_2,min_train_errors_2)



#Grp4,5,7,8 11  are tower one.
filenames_meta_4 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp4_tower_mdl/mdl_tower_mdl_{}/meta_data.yml'
filenames_other_4 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp4_tower_mdl/mdl_tower_mdl_{}/other_data.yml'
files_nb_4 = 58

train_errors_4 = get_train_errors(filenames_meta_4,files_nb_4)
test_errors_4 = get_test_errors(filenames_meta_4, files_nb_4)
min_train_errors_4 = get_min_train_errors(filenames_other_4, files_nb_4)
min_test_errors_4 = get_min_test_errors(filenames_other_4, files_nb_4)
differences_4 = get_difference(test_errors_4,train_errors_4)
differences_of_min_4 = get_difference(min_test_errors_4,min_train_errors_4)

filenames_meta_5 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp5_tower_fixed_mdl/mdl_tower_fix_acti_mdl_{}/meta_data.yml'
filenames_other_5 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp5_tower_fixed_mdl/mdl_tower_fix_acti_mdl_{}/other_data.yml'
files_nb_5 = 362

train_errors_5 = get_train_errors(filenames_meta_5,files_nb_5)
test_errors_5 = get_test_errors(filenames_meta_5, files_nb_5)
min_train_errors_5 = get_min_train_errors(filenames_other_5, files_nb_5)
min_test_errors_5 = get_min_test_errors(filenames_other_5, files_nb_5)
differences_5 = get_difference(test_errors_5,train_errors_5)
differences_of_min_5 = get_difference(min_test_errors_5,min_train_errors_5)


filenames_meta_7 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp7_tower_mdl/mdl_tower_mdl_{}/meta_data.yml'
filenames_other_7 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp7_tower_mdl/mdl_tower_mdl_{}/other_data.yml'
files_nb_7 = 37

train_errors_7 = get_train_errors(filenames_meta_7,files_nb_7)
test_errors_7 = get_test_errors(filenames_meta_7, files_nb_7)
min_train_errors_7 = get_min_train_errors(filenames_other_7, files_nb_7)
min_test_errors_7 = get_min_test_errors(filenames_other_7, files_nb_7)
differences_7 = get_difference(test_errors_7,train_errors_7)
differences_of_min_7 = get_difference(min_test_errors_7,min_train_errors_7)

filenames_meta_8 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp8_tower_mdl/mdl_tower_mdl_{}/meta_data.yml'
filenames_other_8 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp8_tower_mdl/mdl_tower_mdl_{}/other_data.yml'
files_nb_8 = 258

train_errors_8 = get_train_errors(filenames_meta_8,files_nb_8)
test_errors_8 = get_test_errors(filenames_meta_8, files_nb_8)
min_train_errors_8 = get_min_train_errors(filenames_other_8, files_nb_8)
min_test_errors_8 = get_min_test_errors(filenames_other_8, files_nb_8)
differences_8 = get_difference(test_errors_8,train_errors_8)
differences_of_min_8 = get_difference(min_test_errors_8,min_train_errors_8)

filenames_meta_11 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp11_tower_mdl/mdl_tower_mdl_{}/meta_data.yml'
filenames_other_11 = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp11_tower_mdl/mdl_tower_mdl_{}/other_data.yml'
files_nb_11 = 406


train_errors_11 = get_train_errors(filenames_meta_11,files_nb_11)
test_errors_11 = get_test_errors(filenames_meta_11, files_nb_11)
min_train_errors_11 = get_min_train_errors(filenames_other_11, files_nb_11)
min_test_errors_11 = get_min_test_errors(filenames_other_11, files_nb_11)
differences_11 = get_difference(test_errors_11,train_errors_11)
differences_of_min_11 = get_difference(min_test_errors_11,min_train_errors_11)


'''
plot the histogram
'''
# train_errors_conv_tower = train_errors_10 + train_errors_9
# test_errors_conv_tower = test_errors_10 + test_errors_9
# min_train_errors_conv_tower = min_train_errors_10 + min_train_errors_9
# min_test_errors_conv_tower = min_test_errors_10 + min_test_errors_9
# differences_conv_tower = differences_10 + differences_9
# differences_of_min_conv_tower = differences_of_min_10 + differences_of_min_9

# train_errors_random = train_errors_3 + train_errors_12 + train_errors_2
# test_errors_random = test_errors_3 + test_errors_12 + test_errors_2
# min_train_errors_random = min_train_errors_3 + min_train_errors_12+ min_train_errors_2
# min_test_errors_random = min_test_errors_3 + min_test_errors_12 + min_test_errors_2
# differences_random = differences_3 + differences_12 + differences_2
# differences_of_min_random = differences_of_min_3 + differences_of_min_12 +differences_of_min_2

# to_plot = [train_errors_conv_tower,test_errors_conv_tower,min_train_errors_conv_tower,min_test_errors_conv_tower,differences_conv_tower,differences_of_min_conv_tower]
# to_plot_name = ['train_errors_conv_tower','test_errors_conv_tower','min_train_errors_conv_tower','min_test_errors_conv_tower','differences_conv_tower','differences_of_min_conv_tower']

# to_plot = [train_errors_random,test_errors_random,min_train_errors_random,min_test_errors_random,differences_random,differences_of_min_random]
# to_plot_name = ['train_errors_random','test_errors_random','min_train_errors_random','min_test_errors_random','differences_random','differences_of_min_random']

# train_errors_tower = train_errors_4 + train_errors_5 +train_errors_7 + train_errors_8 +train_errors_11
# test_errors_tower = test_errors_4 + test_errors_5 + test_errors_7 + test_errors_8+ test_errors_11
# min_train_errors_tower = min_train_errors_4 + min_train_errors_5+ min_train_errors_7+ min_train_errors_8+ min_train_errors_11
# min_test_errors_tower = min_test_errors_4 + min_test_errors_5+ min_test_errors_7+ min_test_errors_8+ min_test_errors_11
# differences_tower = differences_4 + differences_5 +differences_7+differences_8 +differences_11
# differences_of_min_tower = differences_of_min_4 + differences_of_min_5 + differences_of_min_7+ differences_of_min_8+ differences_of_min_11

# to_plot = [train_errors_tower,test_errors_tower,min_train_errors_tower,min_test_errors_tower,differences_tower,differences_of_min_tower]
# to_plot_name = ['train_errors_tower','test_errors_tower','min_train_errors_tower','min_test_errors_tower','differences_tower','differences_of_min_tower']

train_errors_Total = train_errors_3 + train_errors_12 + train_errors_2 +train_errors_4 + train_errors_5 +train_errors_7 + train_errors_8 +train_errors_11 + train_errors_10 + train_errors_9
test_errors_Total = test_errors_3 + test_errors_12 + test_errors_2 +test_errors_4 + test_errors_5 + test_errors_7 + test_errors_8+ test_errors_11 +test_errors_10 + test_errors_9
min_train_errors_Total = min_train_errors_3 + min_train_errors_12+ min_train_errors_2 + min_train_errors_4 + min_train_errors_5+ min_train_errors_7+ min_train_errors_8+ min_train_errors_11 + min_train_errors_10 + min_train_errors_9
min_test_errors_Total = min_test_errors_3 + min_test_errors_12 + min_test_errors_2+ min_test_errors_4 + min_test_errors_5+ min_test_errors_7+ min_test_errors_8+ min_test_errors_11 + min_test_errors_10 + min_test_errors_9
differences_Total = differences_3 + differences_12 + differences_2+differences_4 + differences_5 +differences_7+differences_8 +differences_11 +differences_10 + differences_9
differences_of_min_Total = differences_of_min_3 + differences_of_min_12 +differences_of_min_2 +differences_of_min_4 + differences_of_min_5 + differences_of_min_7+ differences_of_min_8+ differences_of_min_11 +differences_of_min_10 + differences_of_min_9

to_plot = [train_errors_Total,test_errors_Total,min_train_errors_Total,min_test_errors_Total,differences_Total,differences_of_min_Total]
to_plot_name = ['train_errors_Total','test_errors_Total','min_train_errors_Total','min_test_errors_Total','differences_Total','differences_of_min_Total']
#Grp4,5,7,8 11  are tower one.
for i, name in zip(to_plot,to_plot_name):
    plt.hist(i, bins = 200)
    plt.ylabel('Frequency')
    title = name
    plt.xlabel(title)
    plt.title(title)
    plt.savefig(f"/Users/pangda/predicting_generalization/main_full_auto_ml/data/ploting/{title}.png")
    plt.close()
    # plt.show()



'''
plot the performace of one single data point
'''

# epochs_list = [i+1 for i in range(479)] #epoch list
# # print(epochs_list)
# plt.plot(epochs_list, train_errors, 'b')
# plt.plot(epochs_list, test_errors, 'r')
# plt.xlabel('Epochs')
# plt.ylabel('Errors')
# title = 'random_grp2_6'
# plt.title(title)
# plt.show()
# plt.savefig(f'/Users/pangda/predicting_generalization/main_full_auto_ml/data/ploting/{title}.png')
# plt.clf() #clears canvas for new plot
