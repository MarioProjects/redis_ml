FROM python:3.9-rc-buster
RUN pip install --upgrade pip

WORKDIR /code

# Copy requirements file from current directory to file in
# containers code directory we have just created.
COPY requirements_model.txt requirements.txt

# Run and install all required modules in container
RUN pip3 install -r requirements.txt

COPY settings.py .
COPY run_model.py .
COPY helpers.py .
COPY helpers_model.py .
COPY imagenet_classes.txt .

ENV REDIS_HOST="redis_host"

# RUN app.
CMD python run_model.py
