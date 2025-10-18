#!/bin/bash
# Digital Ocean startup script
gunicorn -c gunicorn.conf.py wsgi:application
