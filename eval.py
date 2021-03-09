import logging
import torch
import numpy as np


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    """

    model.eval()
    summ = []
    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:
        data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def evaluate_kd(model, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    """
    model.eval()
    summ = []
    # compute metrics over the dataset
    for i, (data_batch, labels_batch) in enumerate(dataloader):
        data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)

        # compute model output
        output_batch = model(data_batch)
        # loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)
        loss = 0.0  # force validation loss to zero to reduce computation time

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        # summary_batch['loss'] = loss.item()
        summary_batch['loss'] = loss
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean