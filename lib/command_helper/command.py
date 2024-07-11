import argparse
import os
import re

import yaml


class CommandParams:
    def __init__(self, dic):
        self._dic = dic
        # 递归创建Command对象
        for k, v in dic.items():
            if isinstance(v, dict):
                self._dic[k] = CommandParams(v)
            elif isinstance(v, str):
                # str like '1e-4' is float
                # use regex to match float
                # '1e-4'
                reg = r'^[-+]?\d+[eE][-+]\d+$'
                if re.match(reg, v):
                    self._dic[k] = float(v)

    def __getattr__(self, item):
        try:
            return self._dic[item]
        except KeyError:
            return None

    def __setattr__(self, key, value):
        if key == '_dic':
            super(CommandParams, self).__setattr__(key, value)
            return
        self._dic[key] = value

    def __getitem__(self, item):
        try:
            return self._dic[item]
        except KeyError:
            return None

    def __setitem__(self, key, value):
        self._dic[key] = value

    def __delitem__(self, key):
        del self._dic[key]

    def items(self):
        return self._dic.items()

    def get_dict(self):
        new_dic = {}
        for k, v in self._dic.items():
            if isinstance(v, CommandParams):
                new_dic[k] = v.get_dict()
            else:
                new_dic[k] = v
        return new_dic


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        dic = yaml.safe_load(file)
    return CommandParams(dic)


class Command:
    def __init__(self, isTest=False):
        self.args = self.parse_args()
        self.parse_args_config()
        try:
            self.params = read_yaml(self.args.config)
            self.merge_args_to_params()
        except FileNotFoundError:
            raise
        self.get_root_dir()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default='./configs/BUSI.yaml', help="config file path")

        parser.add_argument("--dataset_name", type=str, default=None, help="dataset name")
        parser.add_argument("--dataset_path", type=str, default=None, help="dataset path")

        parser.add_argument("--model_name", type=str, default=None, help="model name")
        parser.add_argument("--classes", type=int, default=None, help="number of classes")
        parser.add_argument("--model_config_path", type=str, default=None,
                            help="model config file path")
        parser.add_argument("--pretrain_weight_path", type=str, default=None, help="pre-trained weight file path")

        parser.add_argument("--loss_function", type=str, default=None, help="loss function name")
        parser.add_argument("--loss_function_config_path", type=str, default=None, help="loss function name")

        parser.add_argument("--optimizer_config_path", type=str, default=None, help="optimizer config file path")

        parser.add_argument("--lr_scheduler_config_path", type=str, default=None, help="lr scheduler config file path")

        parser.add_argument("--run_dir", type=str, default=None, help="run progress based directory")
        parser.add_argument("--need_early_stop", type=bool, default=False, help="need early stop")

        parser.add_argument("--image_size", type=int, default=None, help="Image size")

        parser.add_argument("--result_dir", type=str, default=None,
                            help="Result Root Directory (Only Used For Testing)")

        args = parser.parse_args()
        return args

    def parse_args_config(self):
        change_keys = []
        for key, value in vars(self.args).items():
            if "config_path" in key:
                if value is not None:
                    change_keys.append(key)

        for key in change_keys:
            try:
                setattr(self.args, key.replace("_path", ""), read_yaml(getattr(self.args, key)))
            except FileNotFoundError:
                raise
            delattr(self.args, key)

    def merge_args_to_params(self):
        del self.args.config

        if self.args.image_size is not None:
            self.params.resize_shape = (self.args.image_size, self.args.image_size)

        for key, value in vars(self.args).items():
            if value is not None:
                self.params[key] = value

    def get_root_dir(self):
        if self.params.run_dir is not None:
            run_dir = self.params.run_dir
        else:
            run_dir = "./runs"
        self.params.run_dir = os.path.join(run_dir, self.params.model_name, self.params.dataset_name).__str__()

        if self.params.result_dir is not None:
            result_dir = self.params.result_dir
            self.params.result_dir = os.path.join(result_dir, self.params.model_name,
                                                  self.params.dataset_name).__str__()
