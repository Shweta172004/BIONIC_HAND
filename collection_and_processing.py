import sys
from datetime import datetime
import csv
import time
    # Import SPI library (for hardware SPI) and MCP3008 library.
import Adafruit_GPIO.SPI as SPI
import Adafruit_MCP3008
    # Software SPI configuration:
import numpy as np
import pyemgpipeline as pep
from matplotlib.figure import SubplotParams


#trial_names = ''
#channel_names = ['channel1', 'channel2', 'channel3']
#sample_rate = 1000

CLK  = 18
MISO = 23
MOSI = 24
CS   = 25
mcp = Adafruit_MCP3008.MCP3008(clk=CLK, cs=CS, miso=MISO, mosi=MOSI)
    #Hardware SPI configuration:
    #SPI_PORT   = 0
    #SPI_DEVICE = 0
    #mcp = Adafruit_MCP3008.MCP3008(spi=SPI.SpiDev(SPI_PORT, SPI_DEVICE))

def start_running():
        
        action = input("Enter the action name\n")
           # Initialize MCP3008
        CLK = 18
        MISO = 23
        MOSI = 24
        CS = 25
        channel_names = ['channel1','channel2','channel3']
        mcp = Adafruit_MCP3008.MCP3008(clk=CLK, cs=CS, miso=MISO, mosi=MOSI)
        sample_rate = 1000
        trial_names =''
        print("Data retrieval starts in\n")
        for i in range(5,0,-1):
            print(i)
            time.sleep(1)
        # Define variables
        all_values = []
        for j in range(1000):
            # Read all the ADC channel values in a list.
            values = [0] * 3 #[0]
            timestamp = datetime.now().strftime('%H:%M:%S')
            values = [mcp.read_adc(i) for i in range(3)]
            print(values, "time:", timestamp)
            all_values.append(values)

        data = np.array(all_values)

        max_val = 1023
        min_val = 0
        scaled_values = (data - min_val) / (max_val - min_val) * 2 - 1
        np_array_scaled = (scaled_values + 1) * 1.65 
        centered_values = np_array_scaled - 1.65
        print(centered_values)
        if centered_values is not None:
                #print("NumPy Array:")
                #print(result_array)

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
                # Assuming sample_rate, trial_names, and channel_names are defined appropriately
                np.set_printoptions(precision=8, suppress=True)

                c = pep.wrappers.EMGMeasurement(centered_values, hz=sample_rate, trial_name=trial_names, channel_names=channel_names, emg_plot_params=emg_plot_params)
                # Plot the processed data
                c.plot()
                
                c.apply_dc_offset_remover()
             #   c.plot()
                   
                c.apply_bandpass_filter(bf_order=4, bf_cutoff_fq_lo=20, bf_cutoff_fq_hi=150)
        # Retrieve the filtered data directly from the EMGMeasurement object
             #   c.plot()
                filtered_data = c.data
                #c.plot()  # Plot the filtered data
        print(filtered_data)
        #print(filtered_data.shape)
        #print("\n")
        #print("reshaping to (3,1000)")
        #mod_path = '/content/drive/MyDrive/Amputee_EMG data Collection Filtered _5sec_csv/TCE_data_sumanth/Sumanth_model_file.tflite'
        #mod_path = '/content/drive/MyDrive/Amputee_EMG data Collection Filtered _5sec_csv/TCE_data_sumanth/Sumanth_model_file_quant_f16.tflite
        # Load data from CSV file
        #filtered_data_reshaped = np.reshape(filtered_data, (3, 1000))
        #print(filtered_data_reshaped)
        #print("Data reshaped to :",filtered_data_reshaped.shape)
        # Assuming 'x' is one of the columns in your CSV file
        #x = data.iloc[:, :-1].values
        #y = data.iloc[:,-1].values
        #print(y)
        # Replace 'x' with the actual column name containing your test data
        with open('/home/pi/Desktop/BIONIC-HAND-MAIN-FILE/Right_Hand.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
        # Retrieve the filtered data directly from the EMGMeasurement object
            filtered_data = filtered_data
            #filtered_data = filtered_data.round(3)
        # Check if filtered_data is not None
            if filtered_data is not None:
            # Extract the first column data
                       
                first_column_data =  [channel_data[0] for channel_data in filtered_data] #All 3 channels
                second_column_data = [channel_data[1] for channel_data in filtered_data]
                third_column_data =  [channel_data[2] for channel_data in filtered_data]
                #first_column_data = [channel_data[0] for channel_data in filtered_data]# 1st channel
                
                first_column_data.append(action)
                first_column_data.append("EMG1")
                second_column_data.append(action)
                second_column_data.append("EMG2")             
                third_column_data.append(action)
                third_column_data.append("EMG3")
            
            # Write the first column data horizontally to the CSV file
                csvwriter.writerow(first_column_data)
                csvwriter.writerow(second_column_data)
                csvwriter.writerow(third_column_data)
                
                #csvfile.flush()#
            print("Filtered data appended to CSV file.")
            print("printing the same bandpass processed data in terminal")
            time.sleep(4)
            if filtered_data is not None:
                i = 1
                for channel_data in filtered_data:
                    print(str(i) + " " + str(channel_data))
                    i += 1
        
        #c.plot()

      #  c.apply_full_wave_rectifier()
     #   c.plot()
        
        #c.apply_linear_envelope(le_order=4, le_cutoff_fq=6)
        #c.plot()
        
        #c.apply_end_frame_cutter(n_end_frames=30)
        #c.plot()
        
        #max_amplitude = np.max(np.abs(result_array), axis=0)
        #print('max_amplitude:', max_amplitude)

        
        #c.apply_amplitude_normalizer(max_amplitude)
        #c.plot()
        
       # all_beg_ts = [2.9, 5.6, 0]
       # all_end_ts = [12, 14.5, 999]
        
       # specific_beg_ts = all_beg_ts[0]
       # specific_end_ts = all_end_ts[0]

       # c.apply_segmenter(all_beg_ts[0], all_end_ts[0])
       # c.plot()
      



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




