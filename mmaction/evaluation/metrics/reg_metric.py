# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import pandas as pd
import torch
from mmengine.evaluator import BaseMetric

from mmaction.evaluation import (get_weighted_score, mean_average_precision,
                                 mean_class_accuracy,
                                 mmit_mean_average_precision, top_k_accuracy, plot)
from mmaction.registry import METRICS
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


@METRICS.register_module()
class RegMetric(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     '1/rmse', 'rmse', 'mae', 'ccc'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 dataset: Optional[str] = None,) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in ['1/rmse', 'mae', 'rmse', 'ccc']

        self.metrics = metrics
        self.dataset = dataset

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_score']
            label = data_sample['gt_label']

            pred = pred.squeeze().cpu().numpy()

            result['pred'] = pred
            if label.size(0) == 1:
                # single-label
                result['label'] = label.item()
            else:
                # multi-label
                result['label'] = label.cpu().numpy()

            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        eval_results = dict()
        preds = [x['pred'] for x in results]
        return self.calculate(preds, labels)

    def calculate(self, preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        if self.dataset == "AVEC2014":
            labels = labels[::2]
            pred_1 = [p.item() for i, p in enumerate(preds) if i % 2 == 0]
            pred_2 = [p.item() for i, p in enumerate(preds) if i % 2 != 0]
            preds = [(p1+p2)/2 for p1, p2 in zip(pred_1, pred_2)]
        
        for metric in self.metrics:
            if metric == 'rmse':
                mse = mean_squared_error(labels, preds)
                rmse = np.sqrt(mse)
                eval_results['1/rmse'] = 1. / rmse
                eval_results['rmse'] = rmse

            if metric == 'mae':
                mae = mean_absolute_error(labels, preds)
                eval_results['mae'] = mae

            if metric == 'ccc':
                pcc, ccc = self._concordance_correlation_coefficient(labels, preds)
                eval_results['pcc'] = pcc
                eval_results['ccc'] = ccc

        return eval_results

    def _concordance_correlation_coefficient(self, y_true, y_pred):
        """Concordance correlation coefficient."""
        # Remove NaNs
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred
        })
        df = df.dropna()
        y_true = df['y_true']
        y_pred = df['y_pred']
        # Pearson product-moment correlation coefficients
        cor = pearsonr(y_true, y_pred)[0]
        # Mean
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        # Variance
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        # Standard deviation
        sd_true = np.std(y_true)
        sd_pred = np.std(y_pred)
        # Calculate CCC
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

        ccc = numerator / denominator
        return cor, ccc


@METRICS.register_module()
class PRMetric(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'ccc', 'mse', 'mae', 'acc'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in ['ccc', 'mse', 'mae', 'acc']

        self.metrics = metrics

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_score']
            label = data_sample['gt_label']

            # Ad-hoc for RGBPoseConv3D
            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()

            result['pred'] = pred
            if label.size(0) == 1:
                # single-label
                result['label'] = label.item()
            else:
                # multi-label
                result['label'] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        eval_results = dict()
        preds = [x['pred'] for x in results]
        return self.calculate(preds, labels)

    def calculate(self, preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()

        labels = np.stack(labels, axis=0)
        preds = np.stack(preds, axis=0)
        for metric in self.metrics:
            keys = ['O', 'C', 'E', 'A', 'N']
            if metric == 'ccc':
                ccc_dic = {}
                ccc_sum = 0
                ccc_mean = 0
                pcc_dic = {}
                pcc_sum = 0
                pcc_mean = 0
                for i, key in enumerate(keys):
                    pcc, ccc = self._concordance_correlation_coefficient(labels[:, i], preds[:, i])
                    ccc_dic[key] = np.round(ccc, 6)
                    ccc_sum += ccc
                    ccc_mean = np.round((ccc_sum / len(keys)), 6)

                    pcc_dic[key] = np.round(pcc, 6)
                    pcc_sum += pcc
                    pcc_mean = np.round((pcc_sum / len(keys)), 6)

                ccc_dic['ccc_mean'] = ccc_mean
                pcc_dic['pcc_mean'] = pcc_mean
                eval_results['ccc_mean'] = ccc_mean
                eval_results['ccc'] = ccc_dic
                eval_results['pcc'] = pcc_dic

            if metric == 'mse':
                mse_dic = {}
                mse_sum = 0
                mse_mean = 0
                for i, key in enumerate(keys):
                    res = mean_squared_error(labels[:, i], preds[:, i])
                    mse_dic[key] = np.round(res, 6)
                    mse_sum += res
                    mse_mean = np.round((mse_sum / len(keys)), 6)
                mse_dic['mse_mean'] = mse_mean
                eval_results['mse'] = mse_dic

            if metric == 'mae':
                mae_dic = {}
                mae_sum = 0
                mae_mean = 0
                for i, key in enumerate(keys):
                    res = mean_absolute_error(labels[:, i], preds[:, i])
                    mae_dic[key] = np.round(res, 6)
                    mae_sum += res
                    mae_mean = np.round((mae_sum / len(keys)), 6)
                mae_dic['mae_mean'] = mae_mean
                eval_results['mae'] = mae_dic

            if metric == 'acc':
                mae_dic = {}
                mae_sum = 0
                mae_mean = 0
                for i, key in enumerate(keys):
                    res = 1 - np.abs(labels[:, i] - preds[:, i]).mean()
                    mae_dic[key] = np.round(res, 6)
                    mae_sum += res
                    mae_mean = np.round((mae_sum / len(keys)), 6)
                mae_dic['acc_mean'] = mae_mean
                eval_results['acc'] = mae_dic

        return eval_results

    def _concordance_correlation_coefficient(self, y_true, y_pred):
        """Concordance correlation coefficient."""
        # Remove NaNs
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred
        })
        df = df.dropna()
        y_true = df['y_true']
        y_pred = df['y_pred']
        # Pearson product-moment correlation coefficients
        cor = pearsonr(y_true, y_pred)[0]
        # Mean
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        # Variance
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        # Standard deviation
        sd_true = np.std(y_true)
        sd_pred = np.std(y_pred)
        # Calculate CCC
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

        ccc = numerator / denominator
        return cor, ccc


@METRICS.register_module()
class AffMetric(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'ccc', 'mse', 'mae'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in ['ccc', 'mse', 'mae']

        self.metrics = metrics

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_score']
            label = data_sample['gt_label']

            # Ad-hoc for RGBPoseConv3D
            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()

            result['pred'] = pred
            if label.size(0) == 1:
                # single-label
                result['label'] = label.item()
            else:
                # multi-label
                result['label'] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        eval_results = dict()

        preds = [x['pred'] for x in results]
        return self.calculate(preds, labels)

    def calculate(self, preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()

        labels = np.stack(labels, axis=0)
        preds = np.stack(preds, axis=0)
        for metric in self.metrics:
            keys = ['V', 'A']
            if metric == 'ccc':
                ccc_dic = {}
                ccc_sum = 0
                ccc_mean = 0
                pcc_dic = {}
                pcc_sum = 0
                pcc_mean = 0
                for i, key in enumerate(keys):
                    pcc, ccc = self._concordance_correlation_coefficient(labels[:, i], preds[:, i])
                    ccc_dic[key] = np.round(ccc, 6)
                    ccc_sum += ccc
                    ccc_mean = np.round((ccc_sum / len(keys)), 6)

                    pcc_dic[key] = np.round(pcc, 6)
                    pcc_sum += pcc
                    pcc_mean = np.round((pcc_sum / len(keys)), 6)

                ccc_dic['ccc_mean'] = ccc_mean
                pcc_dic['pcc_mean'] = pcc_mean
                eval_results['ccc_mean'] = ccc_mean
                eval_results['ccc'] = ccc_dic
                eval_results['pcc'] = pcc_dic

            if metric == 'mse':
                mse_dic = {}
                mse_sum = 0
                mse_mean = 0
                for i, key in enumerate(keys):
                    res = mean_squared_error(labels[:, i], preds[:, i])
                    mse_dic[key] = np.round(res, 6)
                    mse_sum += res
                    mse_mean = np.round((mse_sum / len(keys)), 6)
                mse_dic['mse_mean'] = mse_mean
                eval_results['mse'] = mse_dic

            if metric == 'mae':
                mae_dic = {}
                mae_sum = 0
                mae_mean = 0
                for i, key in enumerate(keys):
                    res = mean_absolute_error(labels[:, i], preds[:, i])
                    mae_dic[key] = np.round(res, 6)
                    mae_sum += res
                    mae_mean = np.round((mae_sum / len(keys)), 6)
                mae_dic['mae_mean'] = mae_mean
                eval_results['mae'] = mae_dic

        return eval_results

    def _concordance_correlation_coefficient(self, y_true, y_pred):
        """Concordance correlation coefficient."""
        # Remove NaNs
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred
        })
        df = df.dropna()
        y_true = df['y_true']
        y_pred = df['y_pred']
        # Pearson product-moment correlation coefficients
        cor = pearsonr(y_true, y_pred)[0]
        # Mean
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        # Variance
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        # Standard deviation
        sd_true = np.std(y_true)
        sd_pred = np.std(y_pred)
        # Calculate CCC
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

        ccc = numerator / denominator
        return cor, ccc