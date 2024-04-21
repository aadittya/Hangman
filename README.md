# Hangman
A Hangman Solver using Deep Reinforcement Learning in Tensorflow

## Summary
A corpus of 227300 words in words_250000_train.txt split into 99% train, 0.5% test and 0.5% validation sets was used to train and evaluate the Deep Reinforcement Learning model. The game starts with all letters unknown. An agent plays the game by sequentially guessing letters. Each letter that is guessed is an output of a neural network, the architecture of which is discussed in the next section. Inputs to the neural network include a one-hot encoded representation of the current game state and all the incorrect guesses made so far. Output of the neural network is a single guess letter. For example, if the word to be guessed is "reinforcement", and the agent has already guessed the letters e,i,a,s and t, the current game state is "_ e i _ _ _ _ _ e _ e _ t" and the incorrect guesses are a, s and t respectively. The neural network takes into account sequential information of each known and unknown letter in the current game state via the use of a bidirectional LSTM and an LSTM layer. The Agent plays the game using the neural network for 100 epochs.

## Neural Network
The idea behind the current architecture is the occurence and co-occurence of n-gram patterns that occur frequently in words. In place of hardcoding n-gram based detection, by applying conv1d layer to the one-hot representation of the game state, the model implicitly learns n-gram representations. This is followed by a bidirectional LSTM, self-attention and an LSTM layer to track long range dependencies of patterns in the words. The output of the final LSTM layer is concatenated with a binary representation of the incorrect guesses made so far. The incorrect_guesses input discourages the model from making the same guess twice while also enabling the agent to efficiently explore the action space.  Finally, a 26 neuron softmax layer outputs the guess. The full architecture shown below uses 169,022 trainable parameters and 384 non-trainable parameters. 

![model](https://github.com/aadittya/Hangman/blob/main/model.png)

A custom loss function is used, which adds a custom binary_cross_entropy loss to an auxillary loss term which is a function of the number of incorrect guesses made so far. The model thus penalizes incorrect guesses, the more the incorrect guesses, the more heavily it is penalized.

## Agent
The agent plays the game for a batch of words parallely, using the neural network to guess a letter at each step for each word in the batch. For each epoch the dataset is shuffled to mitigate overfitting. Tensorflow tensors on GPU are used to construct the custom training loop for fast computation. The training loop for each batch of words is as follows:
1. Guess a letter using neural network by running forward prop
2. Update game state for the word
  - if letter guessed is correct, update known letters in the game state and update the one hot representation at the location of the correctly guessed letter in the word
  - if letter guessed is incorrect, update the incorrect_guesses tensor(which is also an input to the model)
3. Add the new game state, incorrect_guesses and other tensors to a cache
4. Continue 1-3, until lives are over for all the words in the batch,(Note: each word is allowed only 6 lives)
5. Use the cache to run backprop on the neural network. 


## Results
Below are the plots of the training_loss, validation_loss and training, validation win rates respectively. 

![loss_vs_epoch](https://github.com/aadittya/Hangman/blob/main/loss_vs_epoch.jpg)

![win_rate_vs_epoch](https://github.com/aadittya/Hangman/blob/main/win_rate_vs_epoch.jpg)

The Agent is able to solve ~55% of words within 6 lives in both the validation set and the testing set. A slight overfitting is observed which suggests room for improvement for the model.

![test_and_validation_loss_win_rate](https://github.com/aadittya/Hangman/blob/main/test_and_validation_loss_win_rate.jpg)

## Notes
- The entire training loop has been constructed using only tensors on GPU, to leverage cuda parallelism and reduce compute time.
- The agent plays a batch of words, instead of one word at a time, significantly reducing compute time
- Adam optimisation with gradient clip norm of 10.0 is used to overcome exploding gradients
"# Hangman" 
