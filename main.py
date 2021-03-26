import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from scipy.optimize import differential_evolution

from realcifar import *


# trainSet = torchvision.datasets.CIFAR10(root='./raw_data', train=True,
#                                         download=True)
# trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4,
#                                           shuffle=True)
#
# testSet = torchvision.datasets.CIFAR10(root='./raw_data', train=False,
#                                        download=True)
# testLoader = torch.utils.data.DataLoader(testSet, batch_size=4,
#                                          shuffle=True)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def conversion(x):
    """Change the RGB value from 0~255 to -1~1, which corresponds to Tensor value"""
    return  x/255*2 -1

def clipping(x):
    """Change the RGB value from -1~1 to 0~1, which allows the image to be showed"""
    return (x+1)/2


def perturb_image(xs, img):
    """

    :param xs: numpy array of perturbation(s), could be one perturbation eg.[16,16,255,255,0] or multiple perturbation eg.[[16,16,255,255,0],[...],...]
    :param img: a numpy array, with pixel value range from -1 to 1
    :return: perturbed image(s), the dimention of the array is 2D, either [[...]] or [[...],[...],...], numpy -1~1
    """
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images
    tile = [len(xs)] + [1] * (xs.ndim + 1)
    imgs = np.tile(img, tile)

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, r, g, b = pixel
            img[0, x_pos, y_pos] = conversion(r)
            img[1, x_pos, y_pos] = conversion(g)
            img[2, x_pos, y_pos] = conversion(b)

    return imgs

def predict_classes(xs, img, target_class, model, minimize=True):
    """

    :param xs: numpy array of perturbation(s), could be one perturbation eg.[16,16,255,255,0] or multiple perturbation eg.[[16,16,255,255,0],[...],...]
    :param img: a numpy array, with pixel value range from -1 to 1
    :param target_class: the target class of the original image
    :param model: the model used to train the dataset
    :param minimize:
    :return: a list of the predicted confidence of the target class
    """
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image(xs, img)
    input = torch.from_numpy(imgs_perturbed).to(device)
    predictions = F.softmax(model(input), dim=1).data.cpu().numpy()[:,target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions

def predict(img, model):
    """
    Predict the confidence of the image that have gone through the model
    img: numpy -1~1

    """
    if img.ndim < 4:
        img = np.array([img])

    img = torch.from_numpy(img)
    img = img.to(device)
    confidence = F.softmax(model(img), dim=1).data.cpu().numpy()
    return confidence


def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False):
    """

    :param x: numpy array of perturbation(s), could be one perturbation eg.[16,16,255,255,0] or multiple perturbation eg.[[16,16,255,255,0],[...],...]
    :param img: a numpy array, with pixel value range from -1 to 1
    :param target_class:
    :param model: the model used to train the dataset
    :param targeted_attack: if the attack is targeted
    :param verbose:
    :return: True or None due to if the attack is success or not
    """
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, img)

    confidence = predict(attack_image,model)[0]
    predicted_class = np.argmax(confidence)
    # print('attacked predicted class: ',classes[predicted_class])

    # If the prediction is what we want (misclassification or
    # targeted classification), return True
    if verbose:
        print('Confidence:', confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        return True
    # NOTE: return None otherwise (not False), due to how Scipy handles its callback function


def attack(img, label, model, target=None, pixel_count=1,
           maxiter=75, popsize=400, verbose=False):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else label

    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0, 32), (0, 32), (0, 256), (0, 256), (0, 256)] * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes(xs, img, target_class,
                               model, target is None)

    def callback_fn(x, convergence):
        return attack_success(x, img, target_class,
                              model, targeted_attack,verbose)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, img)[0]
    prior_probs = predict(img,model)
    predicted_probs = predict(attack_image,model)
    predicted_class = np.argmax(predicted_probs)
    actual_class = label
    success = predicted_class != actual_class
    cdiff = prior_probs[0,actual_class] - predicted_probs[0,actual_class]

    # Show the best attempt at a solution (successful or not)
    # helper.plot_image(attack_image, actual_class, class_names, predicted_class)
    plt.imshow(np.transpose(clipping(attack_image),(1,2,0)))
    plt.show()

    return [pixel_count, actual_class, predicted_class, success, cdiff, prior_probs,
            predicted_probs, attack_result.x]


# def attack_all(models, samples=500, pixels=(1, 3, 5), targeted=False,
#                maxiter=75, popsize=400, verbose=False):
#     results = []
#     for model in models:
#         model_results = []
#         valid_imgs = correct_imgs[correct_imgs.name == model.name].img
#         img_samples = np.random.choice(valid_imgs, samples, replace=False)
#
#         for pixel_count in pixels:
#             for i, img_id in enumerate(img_samples):
#                 print('\n', model.name, '- image', img_id, '-', i + 1, '/', len(img_samples))
#                 targets = [None] if not targeted else range(10)
#
#                 for target in targets:
#                     if targeted:
#                         print('Attacking with target', class_names[target])
#                         if target == y_test[img_id, 0]:
#                             continue
#                     result = attack(img_id, model, target, pixel_count,
#                                     maxiter=maxiter, popsize=popsize,
#                                     verbose=verbose)
#                     model_results.append(result)
#
#         results += model_results
#         helper.checkpoint(results, targeted)
#     return results


if __name__ == '__main__':

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    input = images[0].numpy()  # input = torch.from_numpy(image_perturbed).to(device)
    input = np.array([input])
    input = torch.from_numpy(input)
    input = input.to(device)
    # print(images, labels)
    # print(images[0].shape)
    image = np.transpose(images[0].numpy(),(1,2,0))
    label = labels[0]
    # print(image.shape)
    image = clipping(image)
    # print(image)
    plt.imshow(image)
    plt.show()

    image_id = 99  # Image index in the test set
    pixel = np.array([16, 16, 255, 255, 0])  # pixel = x,y,r,g,b
    first_image = images[0].numpy()
    image_perturbed = perturb_image(pixel,first_image)
    print(image_perturbed.shape)
    a = clipping(image_perturbed[0])
    a = np.transpose(a,(1,2,0))
    plt.imshow(a)
    plt.show()

    # Initialize the model
    model = VGG16().to(device)

    # Load the pretrained model
    pretrained_model = 'cifar_net.pth'
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    model.eval()

    # Calculate the prior_confidence and also confidence after attack
    prior_confidence = predict_classes(np.array([0,0,0,0,0]),first_image,label,model)
    confidence = predict_classes(pixel, first_image, label, model)

    label = label.cpu().data.item()
    print('prior predicted class: ', classes[label])
    print('Prior confidence was', prior_confidence)
    print('Confidence in true class', 'is', confidence)

    success = attack_success(pixel,first_image,label, model)
    print('Attack Success:',success == True)

    # _ = attack(first_image, label, model,target=None, pixel_count=1)
    # print(_)


    # Attack all the images in the test set
    success = 0
    length = 0
    for img, label in test_set:
        length += 1
        img = img.numpy()
        result = attack(img,label,model,target=None,pixel_count=1)
        if result[3] == True:
            success += 1
            print('Success')
        else:
            print('Fail')
    print('Success rate of one pixel attack:',success/length)



