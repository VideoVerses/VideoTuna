import pytorch_lightning as pl
import torch


class TrainBase(pl.LightningModule):
    """
    Base class for training models using PyTorch Lightning.
    This class extends pl.LightningModule and provides a template for
    implementing custom training logic. Users should inherit from this class and
    override the necessary methods to define their training process.
    """

    def __init__(self):
        """
        Initializes the TrainBase class.
        Call the parent class constructor using super().__init__().
        """
        super().__init__()

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers.
        This method should be overridden in the child class to define the
        optimizers and learning rate schedules.
        """
        raise NotImplementedError("Please implement the configure_optimizers method")

    def forward(self):
        """
        Defines the forward pass of the model.
        This method should be overridden in the child class to define the model's
        forward pass.
        """
        raise NotImplementedError("Please implement the forward method")

    def training_step(self, batch, batch_idx):
        """
        Defines a single training step.
        This method should be overridden in the child class to implement the
        logic for a single training step.

        :param batch: A batch of input data.
        :param batch_idx: The index of the current batch.
        :return: A dictionary containing the loss and any other metrics to log.
        """
        raise NotImplementedError("Please implement the training_step method")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Defines a single validation step.
        This method can be overridden in the child class to implement the logic
        for a single validation step.
        If not overridden, it does nothing by default.

        :param batch: A batch of input data.
        :param batch_idx: The index of the current batch.
        :return: A dictionary containing the loss and any other metrics to log.
        """
        pass


"""
# There are another hooks in the LightningModule class that can be overridden
# runs on every device: devices can be GPUs, TPUs, ...
def fit(self):
    configure_callbacks()

    if local_rank == 0:
        prepare_data()

    setup("fit")
    configure_model()
    configure_optimizers()

    on_fit_start()

    # the sanity check runs here

    on_train_start()
    for epoch in epochs:
        fit_loop()
    on_train_end()

    on_fit_end()
    teardown("fit")


def fit_loop():
    torch.set_grad_enabled(True)

    on_train_epoch_start()

    for batch in train_dataloader():
        on_train_batch_start()

        on_before_batch_transfer()
        transfer_batch_to_device()
        on_after_batch_transfer()

        out = training_step()

        on_before_zero_grad()
        optimizer_zero_grad()

        on_before_backward()
        backward()
        on_after_backward()

        on_before_optimizer_step()
        configure_gradient_clipping()
        optimizer_step()

        on_train_batch_end(out, batch, batch_idx)

        if should_check_val:
            val_loop()

    on_train_epoch_end()


def val_loop():
    on_validation_model_eval()  # calls `model.eval()`
    torch.set_grad_enabled(False)

    on_validation_start()
    on_validation_epoch_start()

    for batch_idx, batch in enumerate(val_dataloader()):
        on_validation_batch_start(batch, batch_idx)

        batch = on_before_batch_transfer(batch)
        batch = transfer_batch_to_device(batch)
        batch = on_after_batch_transfer(batch)

        out = validation_step(batch, batch_idx)

        on_validation_batch_end(out, batch, batch_idx)

    on_validation_epoch_end()
    on_validation_end()

    # set up for train
    on_validation_model_train()  # calls `model.train()`
    torch.set_grad_enabled(True)
"""
