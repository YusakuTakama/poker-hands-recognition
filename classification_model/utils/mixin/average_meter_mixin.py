

class GetMetricsDictMixin:

    def get_meters(self):
        raise NotImplementedError

    def get_mode_name(self):
        raise NotImplementedError

    def get_step_metrics_dict(self) -> dict:
        """generate dict of step batch statistics
        for comet_ml.experiment.log_metrics

        Returns:
            dict: dict to be logged by log_metrics
        """
        loss_meter, topk_meter, topk = self.get_meters()
        mode_name = self.get_mode_name()

        metrics_dict = {
            f"{mode_name}_loss_step": loss_meter.value,
        }
        for meter, k in zip(topk_meter, topk):
            metrics_dict[f"{mode_name}_top{k}_step"] = meter.value
        return metrics_dict

    def get_epoch_metrics_dict(self) -> dict:
        """generate dict of epoch avgerage statistics
        for comet_ml.experiment.log_metrics

        Returns:
            dict: dict to be logged by log_metrics
        """
        loss_meter, topk_meter, topk = self.get_meters()
        mode_name = self.get_mode_name()

        metrics_dict = {
            f"{mode_name}_loss_epoch": loss_meter.avg,
        }
        for meter, k in zip(topk_meter, topk):
            metrics_dict[f"{mode_name}_top{k}_epoch"] = meter.avg
        return metrics_dict
