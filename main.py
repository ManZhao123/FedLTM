#!/usr/bin/env python
import copy
import torch
import argparse
import os
import sys
import time
import warnings
import numpy as np
import torchvision
import logging


# Traditional FL
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverltm import FedLtm
# Regularization-based FL

# Model-splitting-based FL

# 这个用不了
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverprox import FedProx
from flcore.servers.serverkd import FedKD
from flcore.servers.serverDynaFed import DynaFed
from flcore.servers.serverdistill import FedDistill
from flcore.servers.servermoon import MOON
# Knowledge-distillation-based FL
from flcore.servers.servergen import FedGen
from flcore.servers.serverftg import FedFTG
from flcore.servers.servertgp import FedTGP
from flcore.servers.serverproto import FedProto
from flcore.servers.servergkd import FedGKD
from flcore.servers.serverfd import FedFD

from flcore.servers.serverfac import FedAC # ############################
# Knowledge-distillation-based pFL


from flcore.trainmodel.models import *

from flcore.trainmodel.resnet import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from utils.utils import ParamDiffAug
logger = logging.getLogger()
# 创建一个日志记录器对象logger,logging.getLogger()返回一个默认的根记录器对象，用于处理日志消息
logger.setLevel(logging.ERROR)
#  设置日志记录器的日志级别为ERROR。这意味着只有ERROR级别的日志消息会被记录，其他级别的日志消息将被忽略。
#  这样可以控制日志记录的详细程度，只记录重要的错误信息。

warnings.simplefilter("ignore")
# 设置警告过滤器，将警告信息忽略掉。通过这行代码，警告消息将不会被显示或打印出来。
# 警告通常用于指示潜在问题或不推荐使用的代码，但有时可能会干扰到程序的执行。
torch.manual_seed(0)


# hyper-params for Text tasks
vocab_size = 98635
max_len=200
emb_dim=32

channel_dict = {
    "Cifar10": 3,
    "CINIC10": 3,
    "Cifar100": 3,
    "Tiny_imagenet":3,
    "mnist": 1,
    "fmnist": 1,
}
imsize_dict = {
    "Cifar10": (32, 32),
    "CINIC10": (32, 32),
    "Cifar100": (32, 32),
    "Tiny_imagenet": (64, 64),
    "mnist": (28, 28),
    "fmnist": (28, 28),
}

def run(args):

    time_list = []
    reporter = MemReporter() # 内存报报告器
    model_str = args.model

    for i in range(args.prev, args.times): # 其实就是训练几轮的意思
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if model_str == "resnet":
            # args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

            '''args.model = torchvision.models.resnet18(pretrained=True).to(args.device)  # fedgkd使用resnet8
            feature_dim = list(args.model.fc.parameters())[0].shape[1]
            args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)'''

            args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

        elif model_str == "cnn": # non-convex
            args.model = ConvNet(num_classes=args.num_classes, dataset=args.dataset,).to(args.device)
            # args.model = CNN(in_features=3, num_classes=args.num_classes).to(args.device)

        elif model_str == "HtFE8":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
            ]

        else:
            raise NotImplementedError


        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedLtm":
            server = FedLtm(args, i)

        elif args.algorithm == "DynaFed":
            server = DynaFed(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)# Linear(in_features=512, out_features=10, bias=True)
            args.model.fc = nn.Identity()# 样做是为了将模型的最后一层全连接层（或称为头部）替换为一个恒等映射，从而将模型的输出直接作为特征表示，而不进行额外的变换或预测。
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "FedDistill":
            server = FedDistill(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    # average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-alpha', "--alpha", type=float, default=0.1, help="non-iid程度，越小代表异构程度越高，但是0代表iid")
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-dataset', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=32, help="local batch size")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=True)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.998)
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.5,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")

    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)


    # FedProx
    parser.add_argument('-mu', "--mu", type=float, default=0.1,
                        help="Proximal rate for FedProx")


    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=5.0)
    parser.add_argument('-weight coefficient', "--wc", type=float, default=5.0)

    # FedDistill
    parser.add_argument('-lam', "--lamda", type=float, default=0.1)

    # FedGen
    # parser.add_argument('-nd', "--noise_dim", type=int, default=256)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=1e-4)
    # parser.add_argument('-hd', "--hidden_dim", type=int, default=8192)
    # parser.add_argument('-se', "--server_epochs", type=int, default=10)  # 这个是在服务器里训练生成器的epoch
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool,
                        default=False)  # 这个参数用来控制特征提取器是否上传到服务器里，False代表上传，True代表不上传，只在本地更新
    # parser.add_argument('-fcd', "--fc_dim", type=int, default=4096,help="对于FedGen/FedFTG/FedMAD的生成器对于噪音和y的全连接层输出维度")

    # FedGKD
    parser.add_argument('-buffer_length', "--buffer_length", type=int, default=5)
    parser.add_argument('-tem', "--temperature", type=float, default=1.0)
    parser.add_argument('-dc', "--distillation_coefficient", type=float, default=0.1)

    # FedTGP
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-lam', "--lamda", type=float, default=0.1)
    parser.add_argument('-mart', "--margin_threthold", type=float, default=100.0)
    parser.add_argument('-se', "--server_epochs", type=int, default=100)



    # practical，模拟真实场景
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="The dropout rate for total clients. 被选中的客户将在每一轮培训中随机退出")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally.一旦一个客户被选为“slow trainers”，它就会比原来的训练器训练得慢")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")


    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Non-IID alpha：{}".format(args.alpha))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    # print("Using DP: {}".format(args.privacy))
    # if args.privacy:
    #     print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch))

    print("------------------synthesis parameters----------------------")
    print("Iteration：{}".format(args.Iteration))
    print("lr_img: {}".format(args.lr_img))
    print("lr_teacher: {}".format(args.lr_teacher))
    print("lr_lr: {}".format(args.lr_lr))
    print("lr_label: {}".format(args.lr_label))
    print("ipc：{}".format(args.ipc))
    print("label_init：{}".format(args.label_init))
    print("expert_epochs：{}".format(args.expert_epochs))
    print("syn_steps：{}".format(args.syn_steps))
    print("trajectories_length：{}".format(args.trajectories_length))
    print("weight_averaging：{}".format(args.weight_averaging))
    print("min_start_epoch：{}".format(args.min_start_epoch))
    print("max_start_epoch：{}".format(args.max_start_epoch))
    print("distill_iter：{}".format(args.distill_iter))
    print("=" * 50)

    run(args)


