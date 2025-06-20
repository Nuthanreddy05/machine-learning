
# Importing required packages and functions
from utils import load_and_prepare_data
from models import GaussianNBModel
from numpy import mean

#Loading dowloaded and extracted data using load_and_prepare_data function from utils
X_train, y_train, X_test, y_test = load_and_prepare_data()

#function to make RGB and Grayscale versions
def ImageVersion(images_input,version):
    input_size = images_input.shape[0]
    if version == "Greyscale":
        images_input = mean(images_input,axis = -1)
    images_output = images_input.reshape(input_size, -1)
    return(images_output)


'''Building GNB model on RGB version'''
RGB_X_train = ImageVersion(X_train,"RGB")
RGB_X_test = ImageVersion(X_test,"RGB")

ModelGNB_RGB = GaussianNBModel()
ModelGNB_RGB.fit(RGB_X_train, y_train)
#evaluation on train
y_pred_train = ModelGNB_RGB.predict(RGB_X_train)
TrainAccuracy_RGB = sum(y_pred_train == y_train)/len(y_train)
#evaluation on test
y_pred_test = ModelGNB_RGB.predict(RGB_X_test)
TestAccuracy_RGB = sum(y_pred_test == y_test)/len(y_test)


'''Building GNB model on Greyscale version'''
Greyscale_X_train = ImageVersion(X_train,"Greyscale")
Greyscale_X_test = ImageVersion(X_test,"Greyscale")

ModelGNB_Greyscale = GaussianNBModel()
ModelGNB_Greyscale.fit(Greyscale_X_train, y_train)
#evaluation on train
y_pred_train = ModelGNB_Greyscale.predict(Greyscale_X_train)
TrainAccuracy_Greyscale = sum(y_pred_train == y_train)/len(y_train)
#evaluation on test
y_pred_test = ModelGNB_Greyscale.predict(Greyscale_X_test)
TestAccuracy_Greyscale = sum(y_pred_test == y_test)/len(y_test)

print("########################################################")
print("Train Accuracy of GNB on RGB Version : ",TrainAccuracy_RGB)
print("Test Accuracy of GNB on RGB Version : ",TestAccuracy_RGB)
print("########################################################")
print("Train Accuracy of GNB on Greyscale Version : ",TrainAccuracy_Greyscale)
print("Test Accuracy of GNB on Greyscale Version : ",TestAccuracy_Greyscale)
print("########################################################")
