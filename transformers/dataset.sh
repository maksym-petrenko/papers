#!/bin/bash

huggingface-cli download FrancophonIA/french-to-english french-to-english-dataset.csv --local-dir .
mv french-to-english-dataset.csv dataset.csv

