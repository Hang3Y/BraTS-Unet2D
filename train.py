import os
import time
import torch
import argparse
import numpy as np

from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from model.UNet import U_Net
from DataLoader import stack_dict_batched, MultiTrainingDataset, MultiTestingDataset
from utils import seed_everything, get_logger, to_device, calc_loss, ConDiceLoss, cal_dihd_percase


np.bool = bool

def parse_args():
    parser = argparse.ArgumentParser()
    "---Related Model---"
    parser.add_argument("--num_classes", type=int, default=3, help="output channel/class of network")

    "---Related other but useful---"
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--batch_size", type=int, default=64, help="train batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument('--lr_scheduler', type=str, default='MultiStepLR', help='lr scheduler')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    "---Related dir---"
    parser.add_argument("--run_name", type=str, default="unet-debug", help="run model name")
    parser.add_argument("--data_path", type=str, default="./data/BraTS2019-precessed/train/train_stack",
                        help="train data path")
    parser.add_argument("--test_data_path", type=str, default="./data/BraTS2019-precessed/test/test_stack",
                        help="test data path")
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")

    parser.add_argument('--seed', type=int, default=13, help='random seed')
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--resume", type=str, default=None, help="load resume")

    args = parser.parse_args()
    return args


def train_main(args, model, loggers):
    loggers.info('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ This is all argsurations^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    for arg, value in vars(args).items():
        loggers.info(f"{arg}: {value}")

    # about data
    train_dataset = MultiTrainingDataset(args.data_path, image_size=args.image_size, mode='train',
                                         requires_name=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    len_train_dataset = len(train_dataset)
    loggers.info(f"Train data cal. :{len_train_dataset}")

    # about model
    model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    maybe_mem = num_params * 4 / 1e6
    loggers.info(f"Number of parameters: {num_params}, Estimated memory usage: {maybe_mem} MB")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(2, args.epochs, 3)),
                                                     gamma=0.9)

    best_loss = 1e10
    for epoch in range(0, args.epochs):
        model.train()
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)

        "---按epoch来训练"
        train_loader = tqdm(train_loader)
        losses_list = []

        ce_loss = CrossEntropyLoss()
        dice_loss = ConDiceLoss(args.num_classes + 1)

        "---根据训练目标选择模型训练"
        for batch, batched_input in enumerate(train_loader):
            batched_input = stack_dict_batched(batched_input)
            batched_input = to_device(batched_input, args.device)

            mask = batched_input["label"]
            mask = mask.squeeze(1)
            mask_pred = model(batched_input["image"])

            loss, loss_ce, loss_dice = calc_loss(pred_mask=mask_pred, gt_mask=mask, ce_loss=ce_loss, dice_loss=dice_loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses_list.append(loss.item())

        scheduler.step()
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr

        avg_epoch_loss = np.mean(losses_list)
        loggers.info(f"epoch:{epoch + 1}, Loss:{avg_epoch_loss:.4f}, lr:{lr}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_path = os.path.join(args.work_dir, "models", args.run_name, f"best_loss_model.pth")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))


def test_main(args, model, loggers):
    loggers.info('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Testing Begain ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    test_dataset = MultiTestingDataset(data_path=args.test_data_path, image_size=args.image_size, mode='test',
                                       requires_name=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    len_test_dataset = len(test_dataset)
    loggers.info(f"Test data cal. :{len_test_dataset}")
    test_pbar = tqdm(test_loader)
    l_testset = len(test_loader)

    metric_list_s = 0.0
    model_saved_path = os.path.join(args.work_dir, "models", args.run_name, f"best_loss_model.pth")
    with open(model_saved_path, "rb") as f:
        print("========= loading model: ", model_saved_path)
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict['model'])

    model.to(args.device)
    model.eval()
    with torch.no_grad():
        for i, batched_input in enumerate(test_pbar):
            batched_input = stack_dict_batched(batched_input)
            batched_input = to_device(batched_input, args.device)

            mask = batched_input["label"]
            mask = mask.squeeze(1)

            mask_pred = model(batched_input["image"])
            out_mask = torch.argmax(torch.softmax(mask_pred, dim=1), dim=1)
            mask_pred_np = out_mask.cpu().detach().numpy()
            mask_np = mask.cpu().detach().numpy()

            metric_list = []
            for mode in range(1, 4):
                if mode == 1:  # 对应 "WT"
                    metric_list.append(cal_dihd_percase(
                        np.logical_or(mask_pred_np == 1, np.logical_or(mask_pred_np == 2, mask_pred_np == 3)),
                        np.logical_or(mask_np == 1, np.logical_or(mask_np == 2, mask_np == 3))))
                elif mode == 2:  # 对应 "TC"
                    metric_list.append(cal_dihd_percase(np.logical_or(mask_pred_np == 2, mask_pred_np == 3),
                                                        np.logical_or(mask_np == 2, mask_np == 3)))
                elif mode == 3:  # 对应 "ET"
                    metric_list.append(cal_dihd_percase(mask_pred_np == 2, mask_np == 2))
            metric_list_s += np.array(metric_list)  # "dice-hd95"

        metric_lists = np.round((metric_list_s / l_testset), decimals=4)
        wt_metric_lists = metric_lists[0]
        tc_metric_lists = metric_lists[1]
        et_metric_lists = metric_lists[2]
        mean_dice = np.round(np.mean(metric_lists, axis=0)[0], 4)
        mean_hd95 = np.round(np.mean(metric_lists, axis=0)[1], 4)

        loggers.info(f"Metric-Dice-HD: {mean_dice, mean_hd95}")
        loggers.info(f"ET-dice-hd95-jac-prec-sens-spec: {et_metric_lists}")
        loggers.info(f"TC-dice-hd95-jac-prec-sens-spec: {tc_metric_lists}")
        loggers.info(f"WT-dice-hd95-jac-prec-sens-spec: {wt_metric_lists}")


if __name__ == '__main__':
    args = parse_args()

    seed_everything(args.seed)
    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}.log"))

    model = U_Net(4, out_ch=4)
    # model = UnetPlusPlus(4, 4, deep_supervision=False)

    train_main(args, model, loggers=loggers)
    test_main(args, model, loggers=loggers)
