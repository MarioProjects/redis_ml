FROM python:3.9-rc-buster
RUN pip install --upgrade pip

WORKDIR /code

# Copy requirements file from current directory to file in
# containers code directory we have just created.
COPY requirements_server.txt requirements.txt

# Run and install all required modules in container
RUN pip3 install -r requirements.txt

COPY settings.py .
COPY run_server.py .
COPY helpers.py .
COPY imagenet_classes.txt .

# Export env variables.
ENV FLASK_RUN_PORT=5000
ENV REDIS_HOST="redis_host"

# RUN app.
CMD gunicorn --workers 4 --bind 0.0.0.0:$FLASK_RUN_PORT run_server:app
