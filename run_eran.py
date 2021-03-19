import sys
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')
from eran import ERAN
from onnx_translator import *
from read_net_file import *
import csv
from config import *

#Run parameters
netname = 'mnist_0.1.onnx'
means= [0.1307]
stds = [0.3081]
epsilon = 0.1
domain = 'deeppoly'


def normalize(image, means, stds):
    # normalization taken out of the network
    for i in range(len(image)):
        image[i] = (image[i] - means[0])/stds[0]


#Read the model
model, is_conv = read_onnx_net(netname)
eran = ERAN(model, is_onnx=True)


#Read the data
filename = '../data/mnist_test.csv'
csvfile = open(filename, 'r')
tests = csv.reader(csvfile, delimiter=',')

correctly_classified_images = 0
verified_images = 0
for i, test in enumerate(tests):
    image= np.float64(test[1:len(test)])/np.float64(255)
    specLB = np.copy(image)
    specUB = np.copy(image)
    #Normalize the input
    normalize(specLB, means, stds)
    normalize(specUB, means, stds)
    #check whether the image is classified correctly
    label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
    print("concrete ", nlb[-1])
    #if yes then try to check for robustness
    if label == int(test[0]):
        correctly_classified_images += 1
        specLB = np.clip(image - epsilon,0,1)
        specUB = np.clip(image + epsilon,0,1)
        normalize(specLB, means, stds)
        normalize(specUB, means, stds)
        perturbed_label, _, nlb, nub,failed_labels, x = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic,label=label)
        print("nlb ", nlb[-1], " nub ", nub[-1],"adv labels ", failed_labels)
        if(perturbed_label==label):
            print("img", i, "Verified", label)
            verified_images += 1
        else:
            print("img", i, "Failed")
    else:
        print("img", i, "Failed")            
    
print('analysis precision ',verified_images,'/ ', correctly_classified_images)
