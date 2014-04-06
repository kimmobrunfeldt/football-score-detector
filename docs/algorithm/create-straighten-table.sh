#!/bin/bash
# Before running this, run count_score in DEBUG mode

cp ../../debug/* .
python burn-text.py large.jpg "1"
python burn-text.py found_blue.jpg "2"
python burn-text.py found_corners.jpg "3"
python burn-text.py lower_long_side.jpg "4"
python burn-text.py large_straight.jpg "5"
convert -delay 100 -loop 0 -resize 400x400 large.jpg found_blue.jpg found_corners.jpg lower_long_side.jpg large_straight.jpg straighten-table.gif
