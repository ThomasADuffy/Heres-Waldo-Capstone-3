# Use an official Python runtime as a parent image
FROM continuumio/anaconda3:latest

# Set the working directory to /app
WORKDIR /

# Copy the current directory contents into the container at /app
ADD . .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "flask_app.py"]