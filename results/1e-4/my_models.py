#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:35:29 2024

@author: saiful
"""
from torchvision import models,transforms
import torch
from torch import optim,nn
from torch.autograd import Variable

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18, resnet34, resnet50, resnet101
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "densenet":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
        
    elif model_name == "ViTForImageClassification":
        print("\n\n\n == ViTForImageClassification ==")

        
        from transformers import ViTForImageClassification, ViTFeatureExtractor
        # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        model_ft = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        set_parameter_requires_grad(model_ft, feature_extract)

        model_ft.fc = nn.Sequential(
                        nn.Linear(768, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, num_classes))
        model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)

        input_size = 224
        
    elif model_name == "ViTForImageClassification2":
        print("\n\n\n == ViTForImageClassification ==")

        
        from transformers import ViTForImageClassification, ViTFeatureExtractor
        # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        model_ft = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features

        model_ft.fc = nn.Sequential(
                        nn.Linear(num_ftrs, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, num_classes))
        model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)

        input_size = 224

    elif model_name == "ConvNextV2ForImageClassification":
        print("\n\n\n == ConvNextV2ForImageClassification ==")

        
        from transformers import ConvNextV2ForImageClassification
        model_ft = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features

        # model_ft.fc = nn.Sequential(
        #                 nn.Linear(num_ftrs, 128),
        #                 nn.ReLU(inplace=True),
        #                 nn.Linear(128, num_classes))
        model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)

        input_size = 224

    elif model_name == "Swinv2ForImageClassification":
        print("\n\n\n == Swinv2ForImageClassification ==")

        
        from transformers import Swinv2ForImageClassification
        model_ft = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features

        # model_ft.fc = nn.Sequential(
        #                 nn.Linear(num_ftrs, 128),
        #                 nn.ReLU(inplace=True),
        #                 nn.Linear(128, num_classes))
        model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)

        input_size = 224
 
    elif model_name == "CvtForImageClassification":
        print("\n\n\n == CvtForImageClassification ==")

        
        from transformers import  CvtForImageClassification
        model_ft = CvtForImageClassification.from_pretrained("microsoft/cvt-13")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features

        # model_ft.fc = nn.Sequential(
        #                 nn.Linear(num_ftrs, 128),
        #                 nn.ReLU(inplace=True),
        #                 nn.Linear(128, num_classes))
        model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)

        input_size = 224
        
    elif model_name == "EfficientFormerForImageClassification":
        print("\n\n\n == EfficientFormerForImageClassification ==")

        from transformers import  EfficientFormerForImageClassification
        model_ft = EfficientFormerForImageClassification.from_pretrained("snap-research/efficientformer-l1-300")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features

        # model_ft.fc = nn.Sequential(
        #                 nn.Linear(num_ftrs, 128),
        #                 nn.ReLU(inplace=True),
        #                 nn.Linear(128, num_classes))
        model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)

        input_size = 224
        
        
    elif model_name == "PvtV2ForImageClassification":
        print("\n\n\n == PvtV2ForImageClassification ==")

        from transformers import  PvtV2ForImageClassification
        model_ft = PvtV2ForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features

        # model_ft.fc = nn.Sequential(
        #                 nn.Linear(num_ftrs, 128),
        #                 nn.ReLU(inplace=True),
        #                 nn.Linear(128, num_classes))
        model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)

        input_size = 224
        
    elif model_name == "MobileViTV2ForImageClassification":
        print("\n\n\n == MobileViTV2ForImageClassification ==")

        from transformers import  MobileViTV2ForImageClassification
        model_ft = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features

        # model_ft.fc = nn.Sequential(
        #                 nn.Linear(num_ftrs, 128),
        #                 nn.ReLU(inplace=True),
        #                 nn.Linear(128, num_classes))
        model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)

        input_size = 224
        
    else:
        print("Invalid model name, exiting...")
        exit()
        
    return model_ft, input_size
    
    