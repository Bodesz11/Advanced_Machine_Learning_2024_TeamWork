import load_data
import network
import torch
from network import TrafficSignEUSpeedLimit
from torchsummary import summary
import cv2



def main():
    # parameters:
    image_size = 128  # 128 is the best
    batch_size = 64
    add_hsv = False  # It is not worth it. Learning is slower, accuracy is more or less the same
    train_a_model = False
    model_save_path = 'best_model_4_FINAL_splits.pth'

    classes = TrafficSignEUSpeedLimit

    # Loading and creating the training data
    train_data_path = '/home/ad.adasworks.com/adrian.bodai/deleteme_regurarly/2_stage_speed_limit_classifier/SimpleCNNClassifier/FINAL_data/train/'# '/home/ad.adasworks.com/adrian.bodai/deleteme_regurarly/2_stage_speed_limit_classifier/'
    # val_data_path = '/home/ad.adasworks.com/adrian.bodai/deleteme_regurarly/2_stage_speed_limit_classifier/validation/'  -> this is for the original validation set
    val_data_path = '/home/ad.adasworks.com/adrian.bodai/deleteme_regurarly/2_stage_speed_limit_classifier/SimpleCNNClassifier/FINAL_data/valid/' # '/home/ad.adasworks.com/adrian.bodai/deleteme_regurarly/2_stage_speed_limit_classifier/pred_image_crops_v2/'   # yolo predicted cropped images
    yolo_predicted = True  # TODO make a better constraint

    validation_data_loader, other_training_data_val = load_data.load_training_data(image_size=image_size,
                                                                                   add_hsv=add_hsv,
                                                                                   batch_size=batch_size,
                                                                                   working_directory=val_data_path,
                                                                                   augmentation_balancing=False,
                                                                                   class_mapper=classes,
                                                                                   yolo_predicted=yolo_predicted)
    # creating the classificator model
    model = network.ClassifierNet(num_classes=len(classes), input_channels=6 if add_hsv else 3)
    print(model)

    if train_a_model:
        train_data_loader, other_training_data = load_data.load_training_data(image_size=image_size,
                                                                              add_hsv=add_hsv,
                                                                              batch_size=batch_size,
                                                                              working_directory=train_data_path,
                                                                              augmentation_balancing=train_a_model,
                                                                              class_mapper=classes,
                                                                              yolo_predicted=yolo_predicted)
        model = network.train_model(model, train_data_loader, validation_data_loader, epochs=100)
    else:
        print(f'Loading back model from path: {model_save_path}')
        # torch.cuda.is_available()
        model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cuda')))
        other_training_data = None
    accuracy = network.evaluate_model(model=model, dataset=validation_data_loader,
                                      other_training_data=other_training_data,
                                      other_validation_data=other_training_data_val,
                                      device='cuda' if train_a_model else 'cpu',
                                      class_mapper=classes)
    print(f"Final Accuracy on Validation Data: {accuracy:.4f}")


if __name__ == '__main__':
    main()
