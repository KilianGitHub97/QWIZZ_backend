bind = "0.0.0.0:443"  # The address and port to bind to
workers = 4  # The number of worker processes
timeout = 120  # The maximum time a worker can be idle before being restarted
loglevel = "info"  # The log level (debug, info, warning, error, critical)

# SSL certificate configuration
certfile = 'fullchain.pem'  # Path to the SSL certificate file
keyfile = 'privkey.pem'  # Path to the private key file

# Maximum number of requests per worker
max_requests = 200  # Maximum number of requests a worker can handle before
# being restarted
max_requests_jitter = 50  # Random value added to the max_requests to
# prevent all workers from restarting at the
# same time
