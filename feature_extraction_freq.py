import numpy as np
from scipy import stats
from scipy.fft import fft
from feature_extraction_functions.feature_extraction_funcs import find_main_freq, spectral_entropy


# Loading data
data = np.load('input_data/data.npy', allow_pickle=True) 
labels = np.load('input_data/labels.npy', allow_pickle=True) 


# Variables

n_obs = labels.size                         # Total number of observations


# Loop to calculate features for each sensor 
# sensor = (1 = Accelerometer, 2 = Gyroscope, 3 = Magnetometer)
for sensor in range (0,6,3):
   
    x_dc_comp = np.zeros(shape= 0)
    y_dc_comp = np.zeros(shape= 0)
    z_dc_comp = np.zeros(shape= 0)
    
    x_main_freq = np.zeros(shape= 0)
    y_main_freq = np.zeros(shape= 0)
    z_main_freq = np.zeros(shape= 0)
    
    x_main_freq_amp = np.zeros(shape= 0)
    y_main_freq_amp = np.zeros(shape= 0)
    z_main_freq_amp = np.zeros(shape= 0)
    
    x_spec_entro = np.zeros(shape = 0)
    y_spec_entro = np.zeros(shape = 0)
    z_spec_entro = np.zeros(shape = 0)
    
    x_med_f = np.zeros(shape= 0)
    y_med_f = np.zeros(shape= 0)
    z_med_f = np.zeros(shape= 0)
    
    x_std_f = np.zeros(shape= 0)
    y_std_f = np.zeros(shape= 0)
    z_std_f = np.zeros(shape= 0)
    
    x_p25_f = np.zeros(shape= 0)
    y_p25_f = np.zeros(shape= 0)
    z_p25_f = np.zeros(shape= 0)
    
    x_p75_f = np.zeros(shape= 0)
    y_p75_f = np.zeros(shape= 0)
    z_p75_f = np.zeros(shape= 0)
    
   
    
    xy_pers_f = np.zeros(shape= 0)
    xz_pers_f = np.zeros(shape= 0)
    yz_pers_f = np.zeros(shape= 0)
    
    xy_xcr_max_f = np.zeros(shape= 0)
    xz_xcr_max_f = np.zeros(shape= 0)
    yz_xcr_max_f = np.zeros(shape= 0)
    
    xy_xcr_std_f = np.zeros(shape= 0)
    xz_xcr_std_f = np.zeros(shape= 0)
    yz_xcr_std_f = np.zeros(shape= 0)
    

# Loop to run through all ovservations
    for i in range(0, n_obs):                # i is the number ob the observation, which is evaluated
       
        x = data[i,sensor]                    # selecting the axes (i): 1-3  Acceleromerter (x,y,z)
        y = data[i,sensor + 1]                #                         4-6  Gyroscope (x,y,z)                    
        z = data[i,sensor + 2]                #                         7-9  Magnetometer (x,y,z)
    
        N = x.size
        f = np.arange(0, 200/2, 200/N)                # frequency vector 

        ## Standardization
        x = (x - np.min(x))/(np.max(x) - np.min(x))
        y = (y - np.min(y))/(np.max(y) - np.min(y))
        z = (z - np.min(z))/(np.max(z) - np.min(z))
        
        x_spec = fft(x)
        y_spec = fft(y)
        z_spec = fft(z)
        
        x_pxx = (2/(N^2))*abs(x_spec)**2
        y_pxx = (2/(N^2))*abs(y_spec)**2
        z_pxx = (2/(N^2))*abs(z_spec)**2
        
        import matplotlib.pyplot as plt
        plt.semilogy(x_pxx[1:N//2], '-b')
        plt.semilogy(x_pxx, '-b')
        
        
        # normalization of power spectrum with total siganl power
        x_pxx = x_pxx/sum(x_pxx)
        y_pxx = y_pxx/sum(y_pxx)
        z_pxx = z_pxx/sum(z_pxx)
        
        # DC component
        x_dc_comp = np.append(x_dc_comp, x_pxx[0])
        y_dc_comp = np.append(y_dc_comp, y_pxx[0])
        z_dc_comp = np.append(z_dc_comp, z_pxx[0])
        
        # Main frequency & main frequency (amplitude)
        x_main_freq = np.append(x_main_freq, find_main_freq(x_pxx[1:N//2], f[1:])[0])
        y_main_freq = np.append(y_main_freq, find_main_freq(y_pxx[1:N//2], f[1:])[0])
        z_main_freq = np.append(z_main_freq, find_main_freq(z_pxx[1:N//2], f[1:])[0])
       
        x_main_freq_amp = np.append(x_main_freq_amp, np.max(x_pxx[1:N//2]))
        y_main_freq_amp = np.append(y_main_freq_amp, np.max(y_pxx[1:N//2]))
        z_main_freq_amp = np.append(z_main_freq_amp, np.max(z_pxx[1:N//2]))


        # spectral entropy
        x_spec_entro = np.append(x_spec_entro, spectral_entropy(x_pxx[1:N//2]))
        y_spec_entro = np.append(y_spec_entro, spectral_entropy(y_pxx[1:N//2]))
        z_spec_entro = np.append(z_spec_entro, spectral_entropy(z_pxx[1:N//2]))


        ## Calculating the features
        # Median
        x_med_f = np.append(x_med_f, np.median(x_pxx))
        y_med_f = np.append(y_med_f, np.median(y_pxx))
        z_med_f = np.append(z_med_f, np.median(z_pxx))
        
        # Standard deviation
        x_std_f = np.append(x_std_f, np.std(x, dtype=np.float64))
        y_std_f = np.append(y_std_f, np.std(y, dtype=np.float64))
        z_std_f = np.append(z_std_f, np.std(z, dtype=np.float64))
        

        # Percentile 25th
        x_p25_f = np.append(x_p25_f , np.percentile(x_pxx, 25))
        y_p25_f = np.append(y_p25_f , np.percentile(y_pxx, 25))
        z_p25_f = np.append(z_p25_f , np.percentile(z_pxx, 25))
        
        # Percentile 75th
        x_p75_f = np.append(x_p75_f , np.percentile(x_pxx, 75))
        y_p75_f = np.append(y_p75_f , np.percentile(y_pxx, 75))
        z_p75_f = np.append(z_p75_f , np.percentile(z_pxx, 75))
        
        
        # Person correlation coefficient
        xy_pers_f = np.append(xy_pers_f, np.corrcoef(x_pxx,y_pxx)[0][1]) 
        xz_pers_f = np.append(xz_pers_f, np.corrcoef(x_pxx,z_pxx)[0][1]) 
        yz_pers_f = np.append(yz_pers_f, np.corrcoef(y_pxx,z_pxx)[0][1]) 
        
        # Cross correlation (maximum)
        xy_xcr_max_f = np.append(xy_xcr_max_f, np.correlate(x_pxx, y_pxx))
        xz_xcr_max_f = np.append(xz_xcr_max_f, np.correlate(x_pxx, z_pxx))
        yz_xcr_max_f = np.append(yz_xcr_max_f, np.correlate(y_pxx, z_pxx))
        
        # Cross correlation (std)
        xy_xcr_std_f = np.append(xy_xcr_std_f, np.std(np.correlate(x_pxx, y_pxx, 'full')))
        xz_xcr_std_f = np.append(xz_xcr_std_f, np.std(np.correlate(x_pxx, z_pxx, 'full')))
        yz_xcr_std_f = np.append(yz_xcr_std_f, np.std(np.correlate(y_pxx, z_pxx, 'full')))
        
    
    features = np.array([x_dc_comp, y_dc_comp, z_dc_comp, \
                         x_main_freq, y_main_freq, z_main_freq, \
                         x_main_freq_amp, y_main_freq_amp, z_main_freq_amp, \
                         x_spec_entro, y_spec_entro, z_spec_entro, \
                         x_med_f, y_med_f, z_med_f, \
                         x_std_f, y_std_f, z_std_f, \
                         x_p25_f, y_p25_f, z_p25_f, \
                         x_p75_f, y_p75_f, z_p75_f, \
                         xy_pers_f, xz_pers_f, yz_pers_f, \
                         xy_xcr_max_f, xz_xcr_max_f, yz_xcr_max_f, \
                         xy_xcr_std_f, xz_xcr_std_f, yz_xcr_std_f ])
        
    
    if sensor == 0:
        acc_freq = features
    elif sensor == 3:
        gyr_freq = features
        
freq_features = np.concatenate((acc_freq, gyr_freq), axis = 0)

# Normalization

# save mean and std before normalization for classification of new data
features_mean_freq = np.mean(freq_features, axis = 1)
features_std_freq = np.std(freq_features, axis = 1)

np.save('extracted_features/features_mean_freq.npy', features_mean_freq ) 
np.save('extracted_features/features_std_freq.npy', features_std_freq ) 

# normalize
# axis 0 is only one sample is nomalized with itself
# axis 1 features are normalized
freq_features_normalized = stats.zscore(freq_features, axis=1)


np.save('extracted_features/freq_features.npy', freq_features_normalized) 
        

        

        