import argparse
import os
import re
import shutil

split_dic = {
    'benign': ['82', '152', '129', '200', '345', '241', '304', '373', '261', '365', '109', '172', '349', '430',
               '133', '164', '309', '3', '108', '364', '99', '237', '305', '169', '17', '190', '153', '322',
               '267', '334', '158', '420', '162', '292', '103', '380', '178', '210', '247', '46', '354', '11',
               '85', '378', '401', '362', '159', '335', '270', '323', '31', '231', '160', '137', '418', '414',
               '86', '403', '395', '287', '91', '194', '356', '182', '301', '272', '177', '7', '306', '243',
               '255', '79', '392', '238', '259', '75', '63', '263', '234', '370', '74', '23', '433', '130',
               '405', '110', '78', '203', '215'],
    'malignant': ['206', '135', '119', '159', '66', '118', '109', '57', '200', '16', '168', '187', '144', '105',
                  '128', '108', '76', '37', '124', '166', '131', '59', '2', '146', '38', '185', '14', '55',
                  '203', '15', '39', '106', '171', '188', '35', '62', '23', '101', '86', '140', '160', '137',
                  '199', '161'],
    'normal': ['68', '32', '113', '28', '109', '53', '35', '15', '42', '6', '97', '96', '80', '43', '99', '8',
               '16', '121', '77', '98', '70', '51', '131']
}


def get_num(file):
    r = "(\w+) ?\((\d+)\)"
    result = re.match(r, file)
    if result is None:
        return None
    return result.group(2)


def generate_images(input_path, output_path):
    # Load the dataset
    all_output_folders = {
        'all': ['benign', 'malignant', 'normal'],
        'bad': ['benign', 'malignant'],
        'benign': ['benign'],
        'malignant': ['malignant'],
        'normal': ['normal']
    }

    for output_folder, folders in all_output_folders.items():
        for folder in folders:
            print(f'Processing {folder} images')
            input_folder = os.path.join(input_path, folder)
            output_folder_path = os.path.join(output_path, f"BUSI_{output_folder}")
            output_folder_path_train = os.path.join(output_folder_path, 'train')
            output_folder_path_train_images = os.path.join(output_folder_path_train, 'images')
            output_folder_path_train_masks = os.path.join(output_folder_path_train, 'masks')
            output_folder_path_val = os.path.join(output_folder_path, 'val')
            output_folder_path_val_images = os.path.join(output_folder_path_val, 'images')
            output_folder_path_val_masks = os.path.join(output_folder_path_val, 'masks')
            os.makedirs(output_folder_path_train_images, exist_ok=True)
            os.makedirs(output_folder_path_train_masks, exist_ok=True)
            os.makedirs(output_folder_path_val_images, exist_ok=True)
            os.makedirs(output_folder_path_val_masks, exist_ok=True)

            all_output_folders = [
                output_folder_path_train_images,
                output_folder_path_val_images,
                output_folder_path_train_masks,
                output_folder_path_val_masks
            ]

            for file in os.listdir(input_folder):
                if not file.endswith('.png'):
                    continue
                index = 0
                if "_" in file:
                    real_name = file.split("_")[0] + ".png"
                    index = 2
                else:
                    real_name = file

                num = get_num(real_name)
                if num is None:
                    continue
                if num in split_dic[folder]:
                    index += 1

                target_folder = all_output_folders[index]
                shutil.copy(os.path.join(input_folder, file), os.path.join(target_folder, real_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None, help="input path")
    parser.add_argument("--output_path", type=str, default=None, help="output path")
    args = parser.parse_args()
    generate_images(args.input_path, args.output_path)
