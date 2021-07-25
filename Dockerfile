FROM python:3.8.10
# Set a directory for the app
WORKDIR /usr/src/whoosh_api
# Copying project to image
COPY . .
# Install python requirements
RUN pip install -r requirements.txt
EXPOSE 5000
# Run Flask-API
CMD ["python", "./main.py"]

