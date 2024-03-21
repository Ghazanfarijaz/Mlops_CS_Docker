# Base image with Python environment
FROM python:3.8

# Create working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Copy pre-trained model
COPY model.pkl /app/model.pkl

# Command to run the model (replace with your script)
CMD ["python", "inference.py"]