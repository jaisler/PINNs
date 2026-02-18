import os
import yaml
import sampling as smp

def main():

    # Configuration file
    with open(r'configuration.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        params = yaml.load(file, Loader=yaml.FullLoader)

    # Check if folder exists
    if not os.path.isdir(params['pathRes']):
        os.makedirs(params['pathRes'], exist_ok=True)

    # Sampling points
    if (params['routine']['sampling']):
        smp.sampling_points(params) 

if __name__ == "__main__":
    main()
