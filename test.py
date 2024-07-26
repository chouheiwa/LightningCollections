import os
import re

from lightning import Trainer

from lib import command_helper, dataloaders, models
from lib import best_model_name

def get_best_model_checkpoint(opt):
    best_path = None
    best_number = -1000
    base_path = os.path.join(opt.run_dir, 'lightning_logs')
    for data in os.listdir(base_path):
        match = re.search(r'\d+', data)
        if match:
            number = int(match.group())
            if number > best_number:
                best_number = number
                best_path = data
    return best_number, os.path.join(base_path, best_path, 'checkpoints', f'{best_model_name}.ckpt'), os.path.join(base_path, best_path, 'checkpoints', f'last.ckpt')

if __name__ == '__main__':
    command = command_helper.Command(isTest=True)

    best_number, best_ckpt_path, last_ckpt_path = get_best_model_checkpoint(command.params)

    command.params.run_dir = os.path.join(command.params.run_dir, 'lightning_logs', f'version_{best_number}')

    test_loader = dataloaders.get_test_data_loader(command.params)

    network = models.get_network_model(command.params, isTrain=False)

    model = models.SimpleImageSegmentationModel(net=network, loss_func=None, opt=command.params)

    trainer = Trainer(
        default_root_dir=command.params.run_dir,
        benchmark=True,
        inference_mode=False
    )
    trainer.test(model, test_loader, ckpt_path=best_ckpt_path)
