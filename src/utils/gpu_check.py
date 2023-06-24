""" This utils module checks if a GPU is available and if not, 
giving the user the choice to continue or not.
"""
import tensorflow as tf

def gpu_check():
    """ Checks if GPU is available
    """
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("WARNING: No GPU detected. The training process may take several hours.")
        user_input = input("INPUT: Do you want to continue without a GPU? (y/n): ")
        if user_input.lower() != 'y':
            raise Exception("Aborting...")
        else:
            print("INFO: Training initiated with CPU only...")
    else:
        print("INFO: GPU detected. Training initiated...")