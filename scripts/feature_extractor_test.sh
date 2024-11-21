#!/bin/bash

# Define the virtual environment directory
VENV_DIR="lacpvenv"
REPOSITORY_PATH="IDS_ONLINE_FILES"
PYTHON_PATH="IDS_ONLINE_FILES/src"
PYTHON_FILE="main.py"
JSON_PATH="IDS_ONLINE_FILES/model_jsons/TOW_IDS_One_Class_test.json"

# Check if the virtual environment exists
if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    python3 -m venv $VENV_DIR
    echo "$VENV_DIR virtual environment not found, creating one..."
fi

# Check if requirements.txt exists
if [ -f "$REPOSITORY_PATH/requirements.txt" ]; then
    cd $REPOSITORY_PATH
    pip install -e .
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Make sure the repository folder is right"
    exit 1
fi

cd ..

if [ -f "$PYTHON_PATH/$PYTHON_FILE" ]; then 
    echo Running python script from $PYTHON_PATH...
    python3 $PYTHON_PATH/$PYTHON_FILE --config "$JSON_PATH"

else 
    echo "Python script not found!"
fi

echo "Environment setup complete."
