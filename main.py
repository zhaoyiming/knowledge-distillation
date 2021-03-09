from torch.utils.data import Dataset, DataLoader
from model import CNN, DenseNet
import torchvision
import torchvision.transforms as transforms
import argparse
import logging
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import utils
from tqdm import tqdm
import numpy as np
from data import data
from eval import evaluate, evaluate_kd

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/cnn_distill',
                    help="Directory containing params.json")


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches
    """

    model.train()
    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # move to GPU if available
            inputs, labels = inputs.to(device), labels.to(device).squeeze()

            # compute model output and loss
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if batch_idx % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                outputs = outputs.data.cpu().numpy()
                labels = labels.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](outputs, labels)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.
    """

    global scheduler
    best_val_acc = 0.0

    # learning rate schedulers for different models:
    if params.model_version == "densenet":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    t0 = time.time()
    for epoch in range(params.num_epochs):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)
        scheduler.step()
        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
    print('{} seconds'.format(time.time() - t0))


# Defining train_kd & train_and_evaluate_kd functions
def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, metrics, params):
    """Train the model on `num_steps` batches
    """

    # set model to training mode
    model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
            output_batch = model(train_batch)

            # get one batch output from teacher_outputs list

            with torch.no_grad():
                output_teacher_batch = teacher_model(train_batch)
            output_teacher_batch = output_teacher_batch.to(device)

            # loss = loss_fn_kd(output_batch, output_teacher_batch, params)
            loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                          loss_fn_kd, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.
    """
    global scheduler
    best_val_acc = 0.0

    # learning rate schedulers for different models:
    if params.model_version == "densenet_distill":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn_distill":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)

    for epoch in range(params.num_epochs):



        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_kd(model, teacher_model, optimizer, loss_fn_kd, train_dataloader,
                 metrics, params)
        scheduler.step()
        # Evaluate for one epoch on validation set
        val_metrics = evaluate_kd(model, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # Set the random seed for reproducible experiments
    random.seed(230)
    torch.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    trainloader, testloader = data()
    train_dl = trainloader
    dev_dl = testloader
    logging.info("- done.")
    if "distill" in params.model_version:
        if params.model_version == "cnn_distill":
            model = CNN.CNN().to(device)
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            loss_fn_kd = CNN.loss_fn_kd
            metrics = CNN.metrics
        elif params.model_version == 'densenet':
            model = DenseNet.DenseNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=1e-4)
            loss_fn_kd = CNN.loss_fn_kd
            metrics = DenseNet.metrics

        if params.teacher == "densenet":
            teacher_model = DenseNet.DenseNet()
            teacher_checkpoint = 'experiments/base_densenet/best.pth.tar'
            teacher_model = teacher_model.to(device)
        # elif params.teacher == "resnet18":
        #     teacher_model = resnet.ResNet18()
        #     teacher_checkpoint = 'experiments/base_resnet18/best.pth.tar'
        #     teacher_model = teacher_model.cuda() if params.cuda else teacher_model

        utils.load_checkpoint(teacher_checkpoint, teacher_model)

        logging.info("Experiment - model version: {}".format(params.model_version))
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        logging.info("First, loading the teacher model and computing its outputs...")
        train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd,
                              metrics, params, args.model_dir)

    # non-KD mode: regular training of the baseline CNN or ResNet-18
    else:
        if params.model_version == "cnn":
            model = CNN.CNN().to(device)
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            loss_fn = CNN.loss_fn
            metrics = CNN.metrics

        elif params.model_version == "densenet":
            model = DenseNet.DenseNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=1e-4)
            loss_fn = DenseNet.loss_fn
            metrics = DenseNet.metrics
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, params,
                           args.model_dir)
