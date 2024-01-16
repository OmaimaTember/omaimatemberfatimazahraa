# omaimatemberfatimazahraa
import cv2
import numpy as np
import imageio
import os

# Function to load PFM file
def load_pfm(file_path):
    return imageio.imread(file_path)

# Function to calculate disparity map
def calculate_disparity(img1, img2):
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img1, img2)
    return disparity

# Main script
main_folder = r'C:\Users\OM\OneDrive\Bureau\data'

# Open a text file for writing
output_file_path = 'output.txt'
with open(output_file_path, 'w') as output_file:

    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)

        if os.path.isdir(folder_path):
            # Read the two images from the subfolder
            img1 = cv2.imread(os.path.join(folder_path, 'im0.png'), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(os.path.join(folder_path, 'im1.png'), cv2.IMREAD_GRAYSCALE)

            # Calculate disparity map
            calculated_disparity = calculate_disparity(img1, img2)

            # Read provided disparity from PFM files disp0.pfm and disp1.pfm
            disparity_files = ['disp0', 'disp1']

            for disparity_file_name in disparity_files:
                disparity_file_path = os.path.join(folder_path, f'{disparity_file_name}.pfm')

                try:
                    provided_disparity = load_pfm(disparity_file_path)

                    # Compare disparities
                    mse = np.mean((calculated_disparity - provided_disparity) ** 2)

                    # Write comparison results to the file
                    output_file.write(f'Comparison for {folder_name}:\n')
                    output_file.write(f'  Disparity File: {disparity_file_name}.pfm\n')
                    output_file.write(f'  Mean Squared Error: {mse}\n\n')

                except FileNotFoundError:
                    output_file.write(f"File {disparity_file_name}.pfm not found in {folder_path}\n")

print(f'Comparison results written to {output_file_path}')

