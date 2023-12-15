FROM python:3.11



# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBUG 0

# Create a non-root user
RUN adduser --disabled-password --gecos '' myuser

# Install build dependencies
RUN apt-get update \
    && apt-get install -y build-essential gcc python3-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/*

#Install pytorch CPU only
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu



# install dependencies
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .

# Set user ownership and permissions for a specific directory (optional)
RUN chown -R myuser:myuser /app

# set work directory
WORKDIR /app

# collect static files
RUN python manage.py collectstatic --noinput

#RUN python manage.py makemigrations --noinput
#RUN python manage.py migrate --noinput

USER myuser
EXPOSE 80
EXPOSE 443

# run gunicorn
#CMD gunicorn --workers=3 --threads=3 --max-requests=20  --certfile=fullchain.pem --keyfile=privkey.pem app.wsgi:application --bind 0.0.0.0:443