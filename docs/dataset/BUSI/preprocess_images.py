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

keep_dic = {
    'benign': ['113', '406', '390', '144', '328', '282', '94', '294', '105', '410', '369', '386', '191', '57', '257',
               '312', '16', '216', '353', '187', '168', '41', '236', '36', '324', '148', '61', '332', '298', '277',
               '77', '220', '20', '2', '125', '426', '308', '165', '427', '132', '431', '124', '348', '21', '221', '76',
               '333', '299', '276', '149', '60', '325', '260', '37', '372', '240', '186', '40', '217', '352', '201',
               '344', '128', '256', '313', '56', '368', '387', '104', '411', '295', '95', '329', '283', '145', '391',
               '112', '407', '375', '230', '119', '30', '288', '67', '88', '271', '71', '363', '226', '26', '318',
               '174', '436', '123', '359', '135', '115', '400', '379', '396', '142', '84', '284', '154', '338', '416',
               '197', '51', '251', '314', '8', '206', '343', '139', '10', '355', '47', '302', '246', '303', '180',
               '211', '138', '207', '342', '9', '250', '196', '179', '50', '381', '102', '417', '293', '339', '155',
               '285', '143', '397', '114', '421', '134', '358', '5', '437', '122', '175', '319', '27', '227', '70',
               '66', '89', '289', '266', '118', '374', '49', '249', '422', '121', '434', '6', '208', '176', '199', '24',
               '224', '361', '73', '273', '336', '65', '265', '320', '32', '232', '398', '377', '300', '245', '45',
               '183', '357', '212', '12', '341', '204', '316', '253', '53', '228', '382', '28', '101', '290', '90',
               '156', '269', '286', '69', '140', '394', '402', '117', '116', '87', '68', '141', '268', '157', '291',
               '415', '29', '229', '383', '52', '317', '252', '340', '205', '13', '213', '44', '244', '233', '399',
               '376', '33', '264', '321', '64', '337', '72', '225', '360', '419', '198', '209', '120', '435', '136',
               '423', '248', '48', '161', '43', '185', '351', '214', '14', '428', '347', '202', '310', '55', '193',
               '384', '412', '107', '279', '296', '96', '150', '280', '80', '146', '404', '38', '111', '166', '189',
               '18', '131', '218', '127', '432', '59', '170', '22', '388', '222', '367', '275', '330', '326', '408',
               '34', '371', '235', '409', '35', '262', '327', '62', '274', '331', '389', '223', '366', '258', '171',
               '126', '1', '219', '19', '425', '167', '188', '39', '393', '239', '81', '147', '281', '97', '151', '278',
               '297', '413', '106', '385', '192', '311', '254', '15', '429', '350', '42', '184', '307', '242'],
    'normal': ['52', '13', '108', '44', '124', '87', '91', '29', '132', '112', '104', '48', '128', '33', '64', '72',
               '25', '24', '73', '65', '129', '49', '105', '133', '90', '86', '69', '125', '1', '45', '12', '58', '114',
               '102', '19', '62', '74', '23', '54', '118', '39', '122', '81', '78', '79', '123', '38', '7', '14', '119',
               '55', '22', '75', '63', '34', '18', '103', '115', '59', '21', '76', '60', '37', '100', '116', '95', '83',
               '120', '4', '40', '17', '56', '57', '41', '5', '82', '94', '117', '101', '36', '9', '61', '20', '130',
               '93', '85', '126', '2', '46', '11', '50', '27', '66', '89', '31', '106', '110', '111', '107', '30', '67',
               '88', '71', '26', '10', '47', '3', '127', '84', '92'],
    'malignant': ['178', '51', '197', '139', '10', '210', '181', '47', '115', '142', '84', '6', '92', '154', '103',
                  '174', '123', '162', '30', '67', '88', '71', '158', '26', '27', '70', '89', '31', '163', '134', '122',
                  '175', '102', '7', '93', '155', '143', '85', '114', '180', '46', '138', '11', '207', '179', '50',
                  '196', '172', '125', '133', '164', '36', '61', '148', '77', '98', '20', '191', '129', '41', '113',
                  '82', '152', '94', '104', '1', '153', '95', '83', '145', '112', '169', '40', '186', '17', '201',
                  '190', '56', '21', '99', '60', '149', '165', '132', '173', '22', '75', '63', '34', '189', '18', '127',
                  '170', '107', '96', '150', '79', '80', '111', '43', '202', '193', '54', '192', '184', '42', '110',
                  '147', '81', '97', '151', '3', '78', '58', '126', '19', '130', '167', '74', '28', '156', '90', '4',
                  '69', '117', '45', '183', '12', '204', '195', '24', '8', '73', '65', '32', '49', '121', '208', '176',
                  '198', '177', '209', '120', '136', '48', '33', '64', '9', '72', '25', '194', '52', '205', '13', '44',
                  '182', '116', '87', '141', '68', '157', '91', '5', '100', '29']
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
                if "_" in file:
                    real_name = file.split("_")[0] + ".png"
                else:
                    real_name = file
                num = get_num(real_name)
                if num is None:
                    continue
                is_train = num in keep_dic[folder]
                is_val = num in split_dic[folder]

                if not is_train and not is_val:
                    continue

                index = 0 if is_train else 1

                if "_" in file:
                    real_name = file.split("_")[0] + ".png"
                    index += 2
                else:
                    real_name = file


                target_folder = all_output_folders[index]
                shutil.copy(os.path.join(input_folder, file), os.path.join(target_folder, real_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None, help="input path")
    parser.add_argument("--output_path", type=str, default=None, help="output path")
    args = parser.parse_args()
    generate_images(args.input_path, args.output_path)
