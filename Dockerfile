# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirement.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV key venv

# Run app.py when the container launches
CMD ["python", "app.py"]
