#!/bin/bash -l
#SBATCH --job-name=portfolio_opt
#SBATCH --output=portfolio_opt_%A_%a.out
#SBATCH --error=portfolio_opt_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4000MB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-99

echo "Running job array ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Starting at $(date)"

# Path to your manually created virtual environment
VENV_DIR="$WRKDIR/quantum-portfolio/venv"

# Activate the pre-existing virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment at ${VENV_DIR}"
    source $VENV_DIR/bin/activate
else
    echo "Error: Virtual environment not found at ${VENV_DIR}"
    echo "Please create it manually using:"
    echo "mkdir -p $WRKDIR/quantum-portfolio"
    echo "python -m venv $VENV_DIR"
    echo "source $VENV_DIR/bin/activate"
    echo "pip install numpy scipy yfinance pycma pennylane PyPortfolioOpt"
    exit 1
fi

# Set total number of batches
TOTAL_BATCHES=100

# Run the script
python experiments.py ${SLURM_ARRAY_TASK_ID} ${TOTAL_BATCHES}

echo "Job finished at $(date)"
