import pandas as pd
import ast
import cv2
import os
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.transform import rotate
import numpy as np
from collections import Counter
from tqdm import tqdm
from network import TrafficSignEUSpeedLimit
import matplotlib.pyplot as plt
import seaborn as sns


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert label to a tensor if it's not already
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def retype_columns(df):
    int_columns = ['width', 'height']
    df[int_columns] = df[int_columns].astype(int)
    float_columns = ['BoundingBox3D Origin X', 'BoundingBox3D Origin Y', 'BoundingBox3D Origin Z',
                     'BoundingBox3D Extent X', 'BoundingBox3D Extent Y', 'BoundingBox3D Extent Z']
    df[float_columns] = df[float_columns].astype(float)
    return df


def make_df_from_csv(csv_path):
    df = pd.read_csv(csv_path, sep=',')

    def add_object_type_as_seperate_column(row):
        return ast.literal_eval(row['ObjectMeta'])['SubType']

    def add_occluded(row):
        return max([float(value) for key, value in ast.literal_eval(row['Occluded']).items()])

    def add_id_column(row):
        return f'{row["section_id"]}_{row["frame_id"]}_{row["ObjectId"]}'
    df['SubType'] = df.apply(add_object_type_as_seperate_column, axis=1)
    df['sec_frame_obj_id'] = df.apply(add_id_column, axis=1)
    df['Occluded'] = df.apply(add_occluded, axis=1)
    return retype_columns(df)


def filter_df(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    query_parts = []
    for col, conditions in filters.items():
        if isinstance(conditions[0], tuple):  # If there are multiple conditions for the same column
            for op, val in conditions:
                query_parts.append(f'{col} {op} {val}')
        else:  # Single condition
            op, val = conditions
            query_parts.append(f'{col} {op} {val}')

    query_str = ' & '.join(query_parts)
    return df.query(query_str)


def load_npz_based(working_dir: str):
    # Loading gt dataframe
    df = make_df_from_csv(os.path.join(working_dir, "crop_boxes.csv"))
    # Loading gt pictures
    npz_path = os.path.join(working_dir, 'crop_imgs.npz')
    images = np.load(npz_path)

    filters = {
        'Occluded': [('>', 0), ('<', 0.05)],
        '`BoundingBox3D Origin X`': ('<', 60),
        'sec_frame_obj_id': ('in', str(list(images.keys())))
    }
    # Only keeping the relevant lines
    df_relevant = filter_df(df, filters)
    # only keeping the pictures which are relevant
    images = [images[x] for x in df_relevant['sec_frame_obj_id'].to_list()]
    targets = [x for x in df_relevant['SubType'].to_list()]

    # Images are returned in BGR format
    return images, targets, df_relevant


def create_data_loader(x_train, y_train_encoded, batch_size, shuffle, input_channels=3):
    """
    Create a data loader for the given dataset.

    Parameters:
    - x_train: List or array of images
    - y_train_encoded: List or array of encoded labels
    - batch_size: Batch size for the data loader
    - shuffle: Boolean indicating whether to shuffle the data
    - input_channels: Number of input channels (3 for RGB, 6 for RGB + HSV)

    Returns:
    - DataLoader object
    """
    # Define any image transformations
    if input_channels == 6:
        # Normalize with different means and stds if needed
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5],  # Example mean for 6 channels
                                 std=[0.229, 0.224, 0.225, 0.5, 0.5, 0.5])   # Example std for 6 channels
        ])
    else:
        # Default normalization for RGB images
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_dataset = ImageDataset(x_train, y_train_encoded, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

def load_yolo_based(working_dir):
    labels = os.listdir(working_dir)

    x_train = []
    y_train = []
    for label in labels:
        directory = os.path.join(working_dir, label)
        for img_name in os.listdir(directory):
            img_path = os.path.join(directory, img_name)
            img = cv2.imread(img_path)
            x_train.append(img)
            if 'traffic_sign_' in label:
                y_train.append(label[13:])
            else:
                y_train.append(label)
    other_data = pd.DataFrame(index=range(len(x_train)))
    return x_train, y_train, other_data

def load_folder_based(working_directory):
    raise NotImplemented

def load_training_data(image_size: int = 64, add_hsv: bool = False, batch_size=32,
                       augmentation_balancing=True,
                       working_directory='', class_mapper=TrafficSignEUSpeedLimit, yolo_predicted=False) -> tuple:
    print(f'Loading data from {working_directory}')

    # Loading the npz based training data
    if yolo_predicted:
        x_train, y_train, other_data = load_yolo_based(working_directory)
    elif 'crop_boxes.csv' not in os.listdir(working_directory): # folder based images
        x_train, y_train, other_data = load_folder_based(working_directory)
    else:
        x_train, y_train, other_data = load_npz_based(working_directory)

    # Augmenting the data:
    if augmentation_balancing:
        # Optimized for 32 GB of RAM
        augment_max = int(30*((64/image_size)**2)) // (4 if add_hsv else 1)
        x_train, y_train, other_data = augment_balance_data(x_train, y_train, other_data, agumentation_max=augment_max)

    # resizing pictures
    x_train = np.array([cv2.resize(img, (image_size, image_size)) for img in x_train])

    if add_hsv:
        x_train = add_hsv_channel(x_train)

    # Encode string labels to integers
    y_train_encoded = [class_mapper[sign_type.upper()].value for sign_type in y_train]
    other_data['SubType_coded'] = y_train_encoded

    train_data_loader = create_data_loader(x_train, y_train_encoded, batch_size, True,
                                           input_channels=6 if add_hsv else 3)
    print('Training data loaded and processed')
    return train_data_loader, other_data


def add_hsv_channel(x_train):
    x_train_hsv = []

    for img in x_train:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv_img)
        img_with_hsv = np.concatenate((img, h[:, :, np.newaxis], s[:, :, np.newaxis], v[:, :, np.newaxis]), axis=-1)
        x_train_hsv.append(img_with_hsv)
    return np.array(x_train_hsv)


def rotate(image, angle, zoom):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, zoom)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    return rotated


def translate(image, tx, ty, zoom_in):
    (h, w) = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, M, (w, h))
    return zoom(translated, zoom_in=zoom_in)


def zoom(image, zoom_in=1.0):
    (h, w) = image.shape[:2]
    new_w = int(w * zoom_in)
    new_h = int(h * zoom_in)
    resized = cv2.resize(image, (new_w, new_h))
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    zoomed = resized[start_y:start_y + h, start_x:start_x + w]
    return zoomed


def augment_image(image, augment_type):
    if augment_type == 0:
        angle = np.random.uniform(-20, 20)  # np.random.randint(low=-20, high=20)
        return rotate(image, angle=angle, zoom=1 + abs(angle) / 40)
    elif augment_type == 1:
        tx = np.random.randint(-25, 25)
        ty = np.random.randint(-25, 25)
        return translate(image, tx=tx, ty=ty, zoom_in=1 + (0.03 * np.sqrt(tx ** 2 + ty ** 2)))
    elif augment_type == 2:
        zoom_in = np.random.uniform(1.1, 2)
        return zoom(image, zoom_in=zoom_in)


def generate_augmented_images(image, times):
    new_images = []
    for _ in range(times):
        augment_type = np.random.randint(3)
        augmented_image = augment_image(image, augment_type)
        new_images.append(augmented_image)
    return new_images


def plot_class_distribution(y_train_original, y_train_balanced, output_path="class_distribution.png"):
    """
    Plot and save the class distribution before and after augmentation.

    Parameters:
    - y_train_original: Original labels before augmentation
    - y_train_balanced: Labels after augmentation
    - output_path: File path to save the barplot image
    """
    # Count the occurrences of each class in the original and augmented datasets
    original_counts = Counter(y_train_original)
    augmented_counts = Counter(y_train_balanced)

    # Prepare data for plotting
    labels = list(original_counts.keys())
    original = [original_counts[label] for label in labels]
    augmented = [augmented_counts[label] for label in labels]

    # Create a DataFrame for the plot
    plot_data = pd.DataFrame({
        'Class': labels,
        'Original': original,
        'Augmented': augmented
    })

    # Plot the class distribution
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    plot_data = plot_data.melt(id_vars='Class', value_vars=['Original', 'Augmented'],
                               var_name='Dataset', value_name='Count')

    sns.barplot(x='Class', y='Count', hue='Dataset', data=plot_data)

    plt.title('Class Distribution Before and After Augmentation')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)

    # Save the plot to the specified path
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid displaying it

    print(f"Class distribution plot saved to {output_path}")


def plot_augmented_images(x_train_balanced, y_train_balanced, label="eu_speedlimit_110", num_images=8):
    # Get indices of augmented images for the specified label
    indices = [i for i, y in enumerate(y_train_balanced) if y == label]

    # Limit the number of images to plot to `num_images` or total available
    num_images = min(num_images, len(indices))
    if 1 >= num_images:
        return

    # Set up the plot with a grid of 8 images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    fig.suptitle(f"Augmented Images for Label: {label}")

    # Plot each image
    for i, idx in enumerate(indices[:num_images]):
        image = x_train_balanced[idx]  # Retrieve the augmented image
        ax = axes[i]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cmap='gray' if image.ndim == 2 else None)  # Display grayscale if 2D
        ax.axis("off")

    plt.savefig(f'{label}_augmented_images')
    plt.close()


def augment_balance_data(x_train, y_train, other_data, agumentation_max=30):
    print('Augmenting data so that the classes are more balanced')
    label_counts = Counter(y_train)
    max_count = max(label_counts.values())

    x_train_balanced = list(x_train)
    y_train_balanced = list(y_train)

    other_data['augmented'] = False

    for label, count in label_counts.items():
        new_rows = []
        if count < max_count:
            label_indices = [i for i, y in enumerate(y_train) if y == label]
            num_to_augment = min(max_count - count, agumentation_max * count)

            for i in tqdm(range(num_to_augment), desc=f"Augmenting for label {label}", leave=False):
                image_to_augment = x_train[label_indices[i % count]]
                augmented_images = generate_augmented_images(image_to_augment, times=8)

                x_train_balanced.extend(augmented_images)
                y_train_balanced.extend([label] * len(augmented_images))

                plot_augmented_images(augmented_images, [label] * len(augmented_images))

                new_row = other_data.iloc[label_indices[i % count]].copy()
                new_row['augmented'] = True
                new_rows.append(new_row)

            # Convert list of rows to DataFrame and concatenate once per label
            new_rows_df = pd.DataFrame(new_rows)
            other_data = pd.concat([other_data, new_rows_df], axis=0, ignore_index=True)

    plot_class_distribution(y_train, y_train_balanced)

    return x_train_balanced, y_train_balanced, other_data
