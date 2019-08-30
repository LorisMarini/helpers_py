
# --------------------------------- SYSTEM  ---------------------------------

import logging
my_logger = logging.getLogger()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import os                       #
import sys                      #
import mmap                     #
import json                     #
import time                     #
import io                       #
from io import StringIO         #
from os import listdir          #
import shutil                   # To operate with shell-like commands
from shutil import copyfile     # ...
import tempfile                 # Generate temporary files and directories
import tarfile                  # To archive and compress files
import functools                # Higher-order functions and operations on callable objects
import operator                 # Standard operators as functions
import itertools                # Functions creating iterators for efficient looping
from datetime import datetime   #
from datetime import timedelta  #
from datetime import date       #
import argparse                 #
from pprint import pprint       #
from git import Repo            #
import gzip                     #
import glob                     # To search through the filesystem
import hashlib                  # hashlib.algorithms_(see all hash algorithms available)
from hashlib import blake2b     # ...
import inspect                  # To inspect arguments of a function/method
import copy                     # To make shallow and deep copies of objects
import gc                       # Garbage collector
from collections import OrderedDict     # Container OrderedDict
from collections import defaultdict     # Container defaultdict
from typing import List, Set, Dict, Tuple, Text, Optional, AnyStr, Mapping, Any

# ------------------------------ PLOTTING ----------------------------

import matplotlib               # Matplotlib plotting
import seaborn as sns           # R-like plotting

import matplotlib.pyplot as plt # Matplotlib pyplot
matplotlib.use("Agg")           # No plot sin prod

# ------------------------- DATABASES AND ENDPOINTS  ----------------------

import requests         # HTTP library for Python, safe for human consumption.
import pygsheets        # Library to interact with Google Sheets
import s3fs             # Pythonic wrapper to talk to AWS S3
import gcsfs            # Pythonic wrapper to talk to Google GCS
import boto3            # Amazon Web Services Software Development Kit (SDK) for Python
import botocore         # Foundation that underpins Boto 3 (and AWS CLI)
import sqlalchemy       # Python SQL toolkit and Object Relational Mapper

from google.cloud import bigquery
from cloudant import cloudant
from cloudant.view import View
from sqlalchemy.engine.url import URL as sqlalchemy_url
import botocore.exceptions

from pymongo import MongoClient

# ------------------------------ SERIALIZATION ----------------------------

import fastparquet
from fastparquet import write as fpq_write
from fastparquet import ParquetFile
from fastparquet.writer import infer_object_encoding

import pickle

# ------------------------- NUMBERS AND TABLES ------------------------

import pandas as pd
import pandas.util.testing as pdtm
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import pendulum
import numpy as np

from functools import reduce
import random 
import pandas_gbq
from pandas_gbq.schema import generate_bq_schema

# ------------------------- MACHINE LEARNING ------------------------

from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
from sklearn import preprocessing

# ------------------------- AIRFLOW and OPERATORS ------------------------

from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.subdag_operator import SubDagOperator
from airflow.operators.sensors import S3KeySensor
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.contrib.operators.slack_webhook_operator import SlackWebhookOperator
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.hooks.base_hook import BaseHook

# ---------------------------- PARALLELISM  ---------------------------------

import multiprocessing as mp
import subprocess
from subprocess import check_output

# ---------------------------- TESTING  ---------------------------------

import pytest
import pytz

