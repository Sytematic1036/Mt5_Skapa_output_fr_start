#!/bin/bash

# Setup script for Mt5_Skapa_output_fr_start project

echo "Setting up the project..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Creating empty requirements.txt..."
    touch requirements.txt
fi

echo ""
echo "Setup complete!"
echo "To activate the virtual environment manually, run:"
echo "  source venv/Scripts/activate  (Windows/Git Bash)"
echo "  source venv/bin/activate      (Linux/Mac)"
