from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import matplotlib.pyplot as plt

class netTrainer():
    def __init__(self, model, verbose = 0, folder = "model_2B_nUnit"):
        self.model = model
        self.verbose = verbose
        self.filepath = f"{folder}/{model.layers[0].output_shape[1]}.hdf5"
        self.checkpointer = ModelCheckpoint(filepath=self.filepath,
                                            verbose=self.verbose,
                                            monitor='val_mean_squared_error',
                                            mode='min', save_best_only=True)
        self.stopper = EarlyStopping(monitor='val_mean_squared_error', mode='min',
                                     patience=100, verbose = self.verbose)
        self.callbacks = [self.checkpointer, self.stopper]
        self.histories = []

    def fit(self, train_inputs, targets, validation_data, session_params):
        self.train_inputs = train_inputs
        self.train_targets = targets
        self.val_data = validation_data
        num_sessions = len(session_params)
        for sess in range(num_sessions):
            print(f"Training session {sess} of {num_sessions}:")
            print(f"session parameters: {session_params[sess]}")
            self.histories.append(self.single_session(**session_params[sess]))


    def single_session(self, learn_rate, epochs, batch_size=64):
        self.model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=learn_rate, decay=1e-5),
                      metrics =['mean_squared_error', 'mean_absolute_error'])
        history = self.model.fit(self.train_inputs, self.train_targets,
                                 callbacks=self.callbacks, batch_size=batch_size,
                                 epochs = epochs, shuffle=True, verbose=self.verbose,
                                 validation_data=self.val_data)
        return history

    def predict(self, inputs):
        return self.model.predict(inputs)

    def plot_history(self):
        all_val_mses = []
        all_train_mses = []
        for hist in self.histories:
            all_val_mses.extend(hist.history['val_mean_squared_error'])
            all_train_mses.extend(hist.history['mean_squared_error'])
        plt.loglog(all_val_mses, label = 'Validation MSE')
        plt.loglog(all_train_mses, label = 'Training MSE')
        plt.legend()
        plt.show()
