#!/bin/bash

huggingface-cli download FrancophonIA/french-to-english --local-dir . --include french-to-english-dataset.csv
mv french-to-english-dataset.csv dataset.csv

