# Run the Flask app with Gunicorn
# Usage:
#   make run               # local dev using python (honors PORT)
#   make gunicorn          # run under gunicorn using gunicorn.conf.py
#
# Variables:
#   PORT ?= 8080

PORT ?= 8080

.PHONY: run
gunicorn:
	gunicorn -c gunicorn.conf.py wsgi:application

run:
	python wsgi.py
