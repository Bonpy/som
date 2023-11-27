# Self-Organising Map (SOM) Application

This project implements a Self-Organising Map (SOM) for colour categorisation using NumPy. It includes a Jupyter Notebook for exploration and a script for more structured experiments.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook (for running the notebook)
- Docker (for running the application in a container)

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Bonpy/som
cd som
```

Install the required Python packages:

```bash
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Development

### Running the Code Locally

1. **Using Jupyter Notebook:**

    To explore the SOM in an interactive environment, use the Jupyter Notebook:

    - Navigate to the `notebooks` directory:
      ```bash
      cd notebooks
      ```
    - Open the `kohonen.ipynb` notebook and run the cells.

2. **Using Python Script:**

    Before running the script, set up your `PYTHONPATH` to include the directory of your package:

    ```bash
    export PYTHONPATH=$PYTHONPATH:path/to/your/package
    ```

    Then, run the `scripts/train.py` script with desired parameters:

    - Navigate to the `scripts` directory:
      ```bash
      cd scripts
      ```
    - Run the `train.py` script with desired parameters:
      ```bash
      python train.py --colours 20 --iterations 200 --output ./output/result.png
      ```
    - You can adjust the parameters (`--colours`, `--iterations`, `--output`, etc.) as per your requirement.

### Running the Code in a Docker Container

1. **Build the Docker Image:**

    First, build the Docker image from the project root:

    ```bash
    docker build -t som-app .
    ```

2. **Run the Docker Container:**

    Execute the container with the necessary environment variables and volume mount:

    ```bash
    docker run -e COLOURS=20 -e ITERATIONS=200 -e OUTPUT=output/result.png -e GRID_WIDTH=1000 -e GRID_HEIGHT=1000 -e INPUT_DIM=3 -e LR=0.05 -v $(pwd)/output:/usr/src/app/output som-app
    ```

    or on Windows:

    ```
    docker run -e COLOURS=20 -e ITERATIONS=200 -e OUTPUT="output/result.png" -e GRID_WIDTH=200 -e GRID_HEIGHT=200 -e INPUT_DIM=3 -e LR=0.05 -v "//${PWD}/output:/usr/src/app/output" som-app
    ```

    After running, `result.png` will be saved in the `output` directory in your project folder.