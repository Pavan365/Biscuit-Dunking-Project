# Import required libraries.
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

class FNNClassifier(nn.Module):
    """
    Represents a simple Feed-Forward Neural Network for classification. The 
    following structure is used (excluding batch size).

    + Num-Features  : 5
    + Num-Classes   : 3 
    + Dropout       : 0.2

    Input       : [5]
    Dense 1st   : [5]   -> [64]
    ReLU        : [64]  -> [64]
    Dropout     : [64]  -> [64]
    Dense 2nd   : [64]  -> [32]
    ReLU        : [32]  -> [32]
    Dense 3rd   : [32]  -> [16]
    ReLU        : [16]  -> [16]
    Dense 4th   : [16]  -> [3]
    Output      : [3]
    """

    def __init__(self, num_features: int, num_classes: int) -> None:
        """
        Initialises an instance of the FNNClassifier class.

        Parameters
        ----------
        num_features : int
            The number features, used as the input size.

        num_classes : int
            The number of classes, used as the output size.
        """

        # Initialise the parent class.
        super(FNNClassifier, self).__init__()

        # Define the dense layers.
        self.dense_1 = nn.Linear(in_features=num_features, out_features=64)
        self.dense_2 = nn.Linear(in_features=64, out_features=32)
        self.dense_3 = nn.Linear(in_features=32, out_features=16)
        self.dense_4 = nn.Linear(in_features=16, out_features=num_classes)

        # Define the dropout layer.
        self.dropout = nn.Dropout(p=0.2)
        
        # Define the activation functions.
        self.relu = nn.ReLU()
    

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs forward propagation through the neural network.

        Parameters
        ----------
        inputs : torch.Tensor
            The input values for the neural network.
        
        Returns
        -------
        logits : torch.Tensor
            The raw scores from the neural network.
        """

        # Pass through the 1st dense layer.
        logits = self.dropout(self.relu(self.dense_1(inputs)))

        # Pass through the 2nd dense layer.
        logits = self.relu(self.dense_2(logits))

        # Pass through the 3rd dense layer.
        logits = self.relu(self.dense_3(logits))

        # Pass through the 4th dense layer.
        logits = self.dense_4(logits)

        return logits


    def fit(self, 
            train_dataloader: torch.utils.data.DataLoader, 
            val_dataloader: torch.utils.data.DataLoader, 
            epochs: int, 
            learning_rate: float,
            patience: int|None) -> tuple[list]:
        """
        Trains the neural network. This function performs both training and 
        validation.
        
        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            The data loader for the training data.
        
        val_dataloader : torch.utils.data.DataLoader
            The data loader for the validation data.
        
        epochs : int
            The number of epochs to train the neural network over.

        learning_rate : float
            The learning rate for gradient-descent.

        patience : int | None
            The number of patience epochs to wait before early stopping. If set 
            to None, early stopping wont be performed.

        Returns
        -------
        train_losses, val_losses, train_f1s, val_f1s
            A tuple containing lists which contain the train loss, validation 
            loss, train F1 score and validation F1 score for each epoch.
        """

        # Define the loss function. 
        # Define the optimiser.
        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Create lists to store the training and validation loss.
        train_losses = []
        val_losses = []

        # Create lists to store the training and validation F1 score (macro).
        train_f1s = []
        val_f1s = []

        # Create a variable to store the best validation loss.
        # Create a variable to act as a counter for early stopping.
        best_val_loss = np.inf
        stop_counter = 0

        # Perform model training and validation.
        for epoch in range(epochs):
            ## Perform training.
            # Set the model to training mode.
            self.train()

            # Create a variable to store the running training loss.
            # Create a variable to store the running training F1 score.
            train_epoch_loss = 0.0
            train_epoch_f1 = 0.0
            
            # Train on each batch of data.
            for batch_features, batch_targets in train_dataloader:
                # Generate logits (raw scores).
                # Calculate the loss.
                logits = self(batch_features)
                loss = loss_function(logits, batch_targets)

                # Reset the previous gradients.
                optimiser.zero_grad()

                # Calculate gradients using back-propagation.
                # Update model parameters.
                loss.backward()
                optimiser.step()

                # Update the running training loss.
                train_epoch_loss += loss.item()

                # Convert logits (raw scores) to predictions.
                # Calculate the F1 score and update the running training F1 score.
                batch_predictions = torch.argmax(nn.functional.softmax(logits, dim=1), dim=1)
                train_epoch_f1 += f1_score(batch_targets.numpy(), batch_predictions.numpy(), average="macro")

            # Normalise and store the running training loss.
            train_epoch_loss /= len(train_dataloader)
            train_losses.append(train_epoch_loss)
            
            # Normalise and store the running training F1 score.
            train_epoch_f1 /= len(train_dataloader)
            train_f1s.append(train_epoch_f1)

            ## Perform validation.
            # Set the model to evaluation mode.
            self.eval()

            # Create a variable to store the running validation loss.
            val_epoch_loss = 0.0
            val_epoch_f1 = 0.0

            # Disable gradient calculations.
            with torch.no_grad():
                # Validate on each batch of data.
                for batch_features, batch_targets in val_dataloader:
                    # Generate logits (raw scores).
                    # Calculate the loss.
                    logits = self(batch_features)
                    loss = loss_function(logits, batch_targets)

                    # Update the running validation loss.
                    val_epoch_loss += loss.item()

                    # Convert logits (raw scores) to predictions.
                    # Calculate the F1 score.
                    batch_predictions = torch.argmax(nn.functional.softmax(logits, dim=1), dim=1)
                    val_epoch_f1 += f1_score(batch_targets.numpy(), batch_predictions.numpy(), average="macro")

            # Normalise and store the running validation loss.
            val_epoch_loss /= len(val_dataloader)
            val_losses.append(val_epoch_loss)

            # Normalise and store the running validation F1 score.
            val_epoch_f1 /= len(val_dataloader)
            val_f1s.append(val_epoch_f1)

            # Print the epoch and the current metrics.
            print(f"Epoch: {epoch + 1}\n" +
                  f"\tTrain Loss: {train_epoch_loss:.5f}" +
                  f"\tVal Loss  : {val_epoch_loss:.5f}\n" +
                  f"\tTrain F1  : {train_epoch_f1:.5f}" +
                  f"\tVal F1    : {val_epoch_f1:.5f}")
            
            # If a patience was set, check for early stopping.
            if patience is not None:
                # If the validation loss has improved.
                if val_epoch_loss < best_val_loss:
                    # Update the best validation loss.
                    # Reset the stop counter.
                    best_val_loss = val_epoch_loss
                    stop_counter = 0

                # If the validation loss has worsened.
                else:
                    # Increment the stop counter.
                    stop_counter += 1

                # If the stopping criteria has been met.
                if stop_counter >= patience:
                    # Stop the training process.
                    print(f"Stopping Training - Epoch: {epoch + 1}")
                    return train_losses, val_losses, train_f1s, val_f1s

        return train_losses, val_losses, train_f1s, val_f1s


    def predict(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Generates predictions using the neural network.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The data loader for the data to make predictions on.
        
        Returns
        -------
        predictions : numpy.ndarray
            The predictions from the neural network.
        """

        # Set the model to evaluation mode.
        self.eval()

        # Create a list to store all the predictions.
        predictions = []

        # Test on each batch of data.
        for batch in dataloader:
            # Get the batch features.
            batch_features = batch[0]

            # Generate logits (raw scores).
            # Convert logits (raw scores) to predictions.
            logits = self(batch_features)
            batch_predictions = torch.argmax(nn.functional.softmax(logits, dim=1), dim=1)

            # Store the predictions.
            predictions.append(batch_predictions)
        
        # Concatenate the predictions into one NumPy array.
        predictions = torch.cat(predictions).numpy()

        return predictions
