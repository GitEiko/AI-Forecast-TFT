import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import random
import numpy as np
import torch
import threading
import multiprocessing
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def set_seed(seed=42):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def fetch_and_predict(stock_name, max_prediction_length, plot_frame=None):
    try:
        set_seed(512)

        mae = MeanAbsoluteError()
        rmse = MeanSquaredError(squared=False)
        mape = MeanAbsolutePercentageError()

        # Download stock data
        data = yf.download(stock_name, start="2019-06-01", end="2024-01-01")
        if data.empty:
            messagebox.showerror("Error", "Invalid stock ticker or no data available.")
            return

        data = data.reset_index()
        data["Date"] = pd.to_datetime(data["Date"])
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        # Prepare dataset
        data["time_idx"] = range(len(data))
        data["group_id"] = stock_name
        data["value"] = data["Close"]
        max_encoder_length = 2 * max_prediction_length
        last_date = data["Date"].max()
        training_cutoff = last_date - pd.Timedelta(days=(10 * max_prediction_length))

        training = TimeSeriesDataSet(
            data[lambda x: x.Date < training_cutoff],
            time_idx="time_idx",
            target="value",
            group_ids=["group_id"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["group_id"],
            time_varying_known_reals=["Close"],
            time_varying_unknown_reals=["value"],
        )

        min_prediction_idx = data["time_idx"].max() - max_prediction_length + 1
        validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=min_prediction_idx,
                                                    stop_randomization=True)
        batch_size = 128
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="auto",
            gradient_clip_val=0.1,
            limit_train_batches=30,
            callbacks=[lr_logger, early_stop_callback],
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.0007,
            hidden_size=32,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=1,
            loss=MeanSquaredError(),
            log_interval=2,
            reduce_on_plateau_patience=4
        )

        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        future = tft.predict(validation, mode="prediction").detach().numpy()

        # Plot results
        fig = plt.figure(figsize=(12, 6))
        actual_values = data["value"].iloc[-max_prediction_length:].values
        plt.plot(data["Date"].iloc[-max_prediction_length:], actual_values, label="Actual Stock Prices", color="blue",
                 marker='o')
        plt.plot(data["Date"].iloc[-max_prediction_length:], future.flatten(), label="Predicted Stock Prices",
                 color="orange", linestyle="dashed", marker='x')
        plt.title(f"{stock_name} Stock Price Prediction for {max_prediction_length} Days")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()

        # If we have a frame to draw on, use it
        if plot_frame:
            # Clear previous plot if any
            for widget in plot_frame.winfo_children():
                widget.destroy()

            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Metrics calculation continues as before
        y_true = torch.tensor(actual_values)
        y_pred = torch.tensor(future.flatten())

        mae_val = mae(y_pred, y_true)
        rmse_val = rmse(y_pred, y_true)
        mape_val = mape(y_pred, y_true)

        print("MAE:", mae_val.item())
        print("RMSE:", rmse_val.item())
        print("MAPE:", mape_val.item())

        return mae_val.item(), rmse_val.item(), mape_val.item()

    except Exception as e:
        messagebox.showerror("Error", str(e))


# GUI Setup
def run_gui():
    root = tk.Tk()
    root.title("Stock Price Predictor")
    root.geometry("1000x800")  # Make window bigger to fit the plot

    input_frame = tk.Frame(root)
    input_frame.pack(pady=10)

    tk.Label(input_frame, text="Stock Symbol:").grid(row=0, column=0)
    stock_entry = tk.Entry(input_frame)
    stock_entry.grid(row=0, column=1)

    tk.Label(input_frame, text="Prediction Length:").grid(row=1, column=0)
    prediction_entry = tk.Entry(input_frame)
    prediction_entry.grid(row=1, column=1)

    # Create frame for the plot
    plot_frame = tk.Frame(root)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Create frame for metrics
    metrics_frame = tk.Frame(root)
    metrics_frame.pack(fill=tk.X, padx=10, pady=5)

    metrics_label = tk.Label(metrics_frame, text="")
    metrics_label.pack()

    def on_submit():
        stock_name = stock_entry.get().upper()
        try:
            max_prediction_length = int(prediction_entry.get())

            # Create a loading indicator
            loading_label = tk.Label(root, text="Processing prediction, please wait...")
            loading_label.pack(before=plot_frame)
            root.update()

            # Run prediction in a separate thread
            def run_prediction():
                metrics = fetch_and_predict(stock_name, max_prediction_length, plot_frame)
                loading_label.destroy()

                if metrics:
                    mae_val, rmse_val, mape_val = metrics
                    metrics_label.config(text=f"MAE: {mae_val:.4f} | RMSE: {rmse_val:.4f} | MAPE: {mape_val:.4f}")

            thread = threading.Thread(target=run_prediction)
            thread.daemon = True
            thread.start()
        except ValueError:
            messagebox.showerror("Error", "Prediction length must be an integer.")

    submit_button = tk.Button(input_frame, text="Predict", command=on_submit)
    submit_button.grid(row=2, column=0, columnspan=2)

    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_gui()
