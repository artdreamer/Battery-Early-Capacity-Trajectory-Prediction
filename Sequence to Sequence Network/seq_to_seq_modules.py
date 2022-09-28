"""Sequence to sequence modules."""
import random
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import trange


class EncoderLSTM(nn.Module):
    """Encode the input sequence."""

    def __init__(self, input_size, hidden_size, num_layers):
        """
        Initialization.

        :param input_size: the number of features for each element of the sequence, e.g., input embedding size
        :param hidden_size: the number of features of the hidden state, e.g., hidden embedding size
        :param num_layers: the number of stacked LSTM blocks
        """
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)

    def forward(self, inputs):
        """
        cell_size = hidden_size
        output_size = proj_size if proj_size  > 0 else hidden_size
        :param inputs: input sequence [seq_len, batch_size, input_size]
        :return: output: containing the output features (h_t) from the last
                         layer of the LSTM [seq_len, batch_size, output_size]
        :return: h_n: containing the final hidden state for each element
                      in the sequence [num_layers,batch_size,output_size]
        :return: c_n: containing final cell state for each element in the
                      sequence [num_layers,batch_size,cell_size]
        """
        output, (h_n, c_n) = self.lstm(inputs.view(inputs.shape[0], inputs.shape[1], self.input_size))
        return output, (h_n, c_n)


class DecoderLSTMRepeat(nn.Module):
    """
    Decode the output sequence using the repeated input vector from the encoder hidden output
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers, dense_layers, act_func):
        """
        Initialization.
        :param input_size: the number of features for each element of the sequence,
                           e.g., input embedding size
        :param hidden_size: the number of features of the hidden state, e.g., hidden embedding size
        :param output_size: the number of features for each element of the output
        :param num_layers: the number of stacked LSTM blocks
        """
        super(DecoderLSTMRepeat, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        self.dense_layers = nn.Sequential()
        dense_input_size = hidden_size
        for num_layper, num_neurons in enumerate(dense_layers):
            dense_output_size = num_neurons
            self.dense_layers.add_module(
                "layer " + str(num_layper),
                nn.Sequential(nn.Linear(dense_input_size, dense_output_size), act_func)
            )
            dense_input_size = dense_output_size
        self.dense_layers.add_module("output layer", nn.Linear(dense_input_size, output_size))
        # self.dense_layers = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size//2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size//2, output_size),
        # )

    def forward(self, inputs):
        """
        : param x_input:                    should be 3D (seq_len, batch_size, input_size)
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        """
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        output = self.dense_layers(lstm_out.squeeze(0))
        return output, (h_n, c_n)


class DecoderLSTMRecursive(nn.Module):
    """
    Decode the output sequence in a recursive way
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, dense_layers, act_func):
        """
        initialization
        :param input_size: the number of features for each element of the sequence,
                           e.g., input embedding size
        :param hidden_size: the number of features of the hidden state, e.g., hidden embedding size
        :param output_size: the number of features for each element of the output
        :param num_layers: the number of stacked LSTM blocks
        """
        super(DecoderLSTMRecursive, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        self.dense_layers = nn.Sequential()
        dense_input_size = hidden_size
        for num_layper, num_neurons in enumerate(dense_layers):
            dense_output_size = num_neurons
            self.dense_layers.add_module("layer " + str(num_layper),
                                  nn.Sequential(nn.Linear(dense_input_size, dense_output_size), act_func))
            dense_input_size = dense_output_size
        self.dense_layers.add_module("output layer", nn.Linear(dense_input_size, output_size))
        # self.dense_layers = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size//2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size//2, output_size),
        # )

    def forward(self, inputs, encoder_hidden_states):
        """
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        """
        lstm_out, hidden = self.lstm(inputs.unsqueeze(0), encoder_hidden_states)
        output = self.dense_layers(lstm_out.squeeze(0))
        return output, hidden


class SeqToSeqLSTM(nn.Module):
    """
    train LSTM encoder-decoder and make predictions
    """
    def __init__(
            self, input_size, hidden_size, num_layers, decoder_type='repeat_vector',
            device=torch.device("cpu"), dense_layers=[100], act_func=nn.ReLU()
        ):
        """
        :param input_size:
        :param hidden_size:
        :param num_layers:
        """
        super(SeqToSeqLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = EncoderLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder_type = decoder_type
        if decoder_type == 'repeat_vector':
            self.decoder = DecoderLSTMRepeat(
                input_size=hidden_size, hidden_size=hidden_size, output_size=input_size,
                num_layers=num_layers, dense_layers=dense_layers, act_func=act_func
            )
        elif decoder_type == 'recursive':
            self.decoder = DecoderLSTMRecursive(
                input_size=input_size, hidden_size=hidden_size, output_size=input_size,
                num_layers=num_layers, dense_layers=dense_layers, act_func=act_func
            )

        # criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.device = device

    def train_encoder_decoder(self, train_loader, val_loader, n_epochs, batch_size, target_len,
                              training_prediction='recursive', teacher_forcing_ratio=0.5, learning_rate=0.01,
                              gamma=0.9, dynamic_tf=False, early_stopping=False, step_start=10, step_patience=5,
                              monitor_path=None):
        """
        Train lstm encoder-decoder.

        : param train_loader:              training data, input is [seq_len, # * batch_size, input_size] tensor,
                                           target output is [seq_len, # * batch_size, input_size] tensor
        : param val_loader:                validation data
        : param n_epochs:                  number of epochs
        : param batch_size
        : param target_len:                target sequence length
        : param training_prediction:       type of prediction to make during training
        : ('recursive', 'teacher_forcing', or 'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        training_prediction = 'teacher_forcing.' For each batch in training, we generate a random number.
        If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param gamma:                     learning rate decay
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : param early_stopping:            use early stopping if True
        : param step_start:                step # at the start of early stopping check
        : param step_patience:             wait steps after no val metric improvement
        : param monitor_path:              file path to save the model
        : return losses:                   array of loss function for each epoch
        """

        # initialize array of losses
        losses = np.full(n_epochs, np.nan)
        val_losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=30, gamma=gamma)
        size = len(train_loader.dataset)

        if early_stopping:
            early_stop = EarlyStoppingCheck(step_start, step_patience, monitor_path)
            stopping_status = False

        with trange(n_epochs) as tr:
            for it in tr:
                batch_loss = 0.
                for batch_idx, (inputs, target) in enumerate(train_loader):
                    inputs, target = inputs.to(self.device).transpose(0, 1), target.to(self.device).transpose(0, 1)

                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, inputs.shape[2])

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(inputs)

                    if self.decoder_type == 'repeat_vector':
                        decoder_input = encoder_hidden[0][-1, :, :].unsqueeze(0).expand(target_len, -1, -1)
                        outputs, decoder_hidden = self.decoder(decoder_input)
                    elif self.decoder_type == 'recursive':
                        # decoder with teacher forcing
                        decoder_input = inputs[-1, :, :]  # shape: (batch_size, input_size)
                        decoder_hidden = encoder_hidden
                        if training_prediction == 'recursive':
                            # predict recursively
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                        if training_prediction == 'teacher_forcing':
                            # use teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                for t in range(target_len):
                                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                    outputs[t] = decoder_output
                                    decoder_input = target[t, :, :]

                            # predict recursively
                            else:
                                for t in range(target_len):
                                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                    outputs[t] = decoder_output
                                    decoder_input = decoder_output

                        if training_prediction == 'mixed_teacher_forcing':
                            # predict using mixed teacher forcing
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output

                                # predict with teacher forcing
                                if random.random() < teacher_forcing_ratio:
                                    decoder_input = target[t, :, :]

                                # predict recursively
                                else:
                                    decoder_input = decoder_output

                    # compute the loss
                    loss = self.criterion(outputs, target)
                    batch_loss += loss.item()

                    # print loss every 10 batch
                    if batch_idx % 10 == 0:
                        print(
                            "[{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                                batch_idx * batch_size,
                                size,
                                100.0 * batch_idx * batch_size / size,
                                loss.item(),
                            )
                        )

                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # training loss for epoch
                scheduler.step()
                batch_loss *= (batch_size/size)
                losses[it] = batch_loss

                # validation loss for epoch
                _, val_loss = self.predict(val_loader, target_len, batch_size)
                if early_stopping:
                    stopping_status, min_val_metric = early_stop.step(self, val_loss, it)
                if stopping_status:
                    print(f"Early stopping! The validation error has not improved for the past {step_patience} epochs")
                    break
                self.encoder.train()
                self.decoder.train()
                val_losses[it] = val_loss

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02

                # progress bar
                tr.set_postfix(val_loss="{0:.3f}".format(val_losses[it]))
            if not stopping_status:
                print(f"Done! All {n_epochs} training epochs completed!")
                torch.save(self.state_dict(), monitor_path + '/best_model.pt')
        return losses, val_losses

    def predict(self, dataloader, target_len, batch_size):
        self.encoder.eval()
        self.decoder.eval()
        loss = 0
        size = len(dataloader.dataset)
        batch_idx = 1
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(dataloader):
                inputs, target = inputs.to(self.device).transpose(0, 1), target.to(self.device).transpose(0, 1)

                # outputs tensor
                outputs = torch.zeros(target_len, batch_size, inputs.shape[2])

                # encoder outputs
                encoder_output, encoder_hidden = self.encoder(inputs)

                if self.decoder_type == 'repeat_vector':
                    # decode input_tensor
                    decoder_input = encoder_hidden[0][-1, :, :].unsqueeze(0).expand(target_len, -1, -1)
                    outputs, decoder_hidden = self.decoder(decoder_input)
                elif self.decoder_type == 'recursive':
                    # decode input_tensor
                    decoder_input = inputs[-1, :, :]
                    decoder_hidden = encoder_hidden
                    for t in range(target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output.squeeze(0)
                        decoder_input = decoder_output

                loss += self.criterion(outputs, target)

        return outputs, loss  / batch_idx


class EarlyStoppingCheck:
    """Early stopping class."""
    def __init__(self, step_start, step_patience, monitor_path):
        self.step_start = step_start
        self.step_patience = step_patience
        self.early_stop = False
        self.min_val_loss = float("inf")
        self.step_min_val_loss = 0
        self.path = monitor_path + "/best_model.pt"

    def step(self, model, val_loss, epoch):
        """Monitor trianing step."""
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.step_min_val_loss = epoch
            torch.save(model.state_dict(), self.path)
        if epoch < self.step_start:
            return self.early_stop, self.min_val_loss
        elif epoch - self.step_min_val_loss >= self.step_patience:
            self.early_stop = True

        return self.early_stop, self.min_val_loss
