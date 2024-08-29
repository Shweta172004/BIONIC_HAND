import sys
from datetime import datetime
import time
import numpy as np
import Adafruit_GPIO.SPI as SPI
import Adafruit_MCP3008
import pyemgpipeline as pep
from matplotlib.figure import SubplotParams
import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
import pandas as pd
import csv

# Initialize the TFLite interpreter
interpreter = Interpreter(model_path='path of your trained cnn tflite model')
interpreter.allocate_tensors()

# Assuming the input and output details of your TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def amplify_data(filtered_data):
    amplified_data = filtered_data * 2.0  # Example: Amplify by a factor of 2
    return amplified_data

def save_to_csv(data, actual_label, predicted_label):
    csv_file = 'path of your dataset'
    with open(csv_file, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write data to CSV
        csvwriter.writerow(data + [actual_label, predicted_label])

def start_running():
    # Initialize MCP3008
    CLK = 18
    MISO = 23
    MOSI = 24
    CS = 25
    channel_names = ['channel1', 'channel2', 'channel3']
    mcp = Adafruit_MCP3008.MCP3008(clk=CLK, cs=CS, miso=MISO, mosi=MOSI)
    sample_rate = 1000
    trial_names = ''

    print("Data retrieval starts in")
    for i in range(5, 0, -1):
        print(i)
        time.sleep(1)

    # Define variables
    all_values = []
    for j in range(1000):
        # Read all the ADC channel values in a list.
        values = [0] * 3
        timestamp = datetime.now().strftime('%H:%M:%S')
        values = [mcp.read_adc(i) for i in range(3)]  # Read from all three channels
        print(values, "time:", timestamp)
        all_values.append(values)

    data = np.array(all_values)

    max_val = 1023
    min_val = 0
    scaled_values = (data - min_val) / (max_val - min_val) * 2 - 1
    np_array_scaled = (scaled_values + 1) * 1.65
    centered_values = np_array_scaled - 1.65

    if centered_values is not None:
        emg_plot_params = pep.plots.EMGPlotParams(
            n_rows=3,
            fig_kwargs={
                'figsize': (10, 10),
                'dpi': 80,
                'subplotpars': SubplotParams(wspace=0, hspace=0.6),
            },
            line2d_kwargs={
                'color': 'green',
            },
        )
        emg_plot_params.ax_kwargs = {'ylim': (-1.65, 1.65)}

        np.set_printoptions(precision=8, suppress=True)

        c = pep.wrappers.EMGMeasurement(centered_values, hz=sample_rate, trial_name=trial_names,
                                        channel_names=channel_names, emg_plot_params=emg_plot_params)

        c.apply_dc_offset_remover()
        c.apply_bandpass_filter(bf_order=4, bf_cutoff_fq_lo=20, bf_cutoff_fq_hi=150)

        # Amplify the filtered data
        amplified_data = amplify_data(centered_values)
        # Perform inference
        # Adjust NUM_SAMPLES to match your model's input size
        NUM_SAMPLES = 1000
        input_data = amplified_data[:NUM_SAMPLES].reshape((1, NUM_SAMPLES, 3)).astype(np.float32)

        # Set input tensor to the interpreter
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Perform post-processing on output_data if needed
        predicted_class = np.argmax(output_data[0])
        print(f'Predicted action index: {predicted_class}')

        # Print corresponding action
        actions = ["cylindrical", "hook", "lateral", "palmar", "spherical", "tip"]
        if predicted_class < len(actions):
            predicted_label = actions[predicted_class]
            actual_label = "actual_label"  # Replace with your actual label logic
            save_to_csv(amplified_data.flatten().tolist(), actual_label, predicted_label)

        # Clear interpreter's cache
        interpreter.reset_all_variables()

    print("\n")


def main():
    print("Press Enter to start running the script...")

    # Loop until Enter key is pressed
    while True:
        # Read a single character from the standard input without waiting for a newline
        char = sys.stdin.read(1)

        # Check if the character is Enter key (newline)
        if char == '\n':
            # Call the function to start running
            start_running()
            break  # Exit the loop

    # Add a message to prompt the user to press Enter before restarting or terminating
    print("Script finished. Press Enter to restart or press 'q' + Enter to exit...")
    char = input()  # Wait for the user to press Enter before restarting or terminating

    if char.lower() == 'q':
        print("Exiting...")
        sys.exit()
    else:
        main()  # Restart the script

if __name__ == "__main__":
    main()

