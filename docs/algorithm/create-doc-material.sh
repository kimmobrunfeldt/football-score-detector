#!/bin/bash
# Before running this, run count_score in DEBUG mode

# Copy all debug images and resize them to smaller
rm *.jpg
rm *.gif
cp ../../debug/* .
cp ../../testdata/real3.jpg testdata.jpg
mogrify -quality 100 -geometry '500x>' *.jpg

# Create straighten table animation
mkdir giftemp
cp large.jpg found_blue.jpg found_corners.jpg lower_long_side.jpg large_straight.jpg giftemp

python burn-text.py giftemp/large.jpg "1"
python burn-text.py giftemp/found_blue.jpg "2"
python burn-text.py giftemp/found_corners.jpg "3"
python burn-text.py giftemp/lower_long_side.jpg "4"
python burn-text.py giftemp/large_straight.jpg "5"

convert -quality 100 -delay 140 -loop 0 giftemp/large.jpg giftemp/found_blue.jpg giftemp/found_corners.jpg giftemp/lower_long_side.jpg giftemp/large_straight.jpg straighten_table.gif
convert -quality 100 straighten_table.gif \( -clone 0 -set delay 250 \) -swap 0 +delete \( +clone -set delay 250 \) +swap +delete straighten_table.gif

rm -r giftemp

# Create label corners animation
mkdir giftemp
cp corner_a.jpg corner_b.jpg corner_c.jpg corner_d.jpg corner_labels.jpg giftemp

convert -quality 100 -delay 140 -loop 0 giftemp/corner_a.jpg giftemp/corner_b.jpg giftemp/corner_c.jpg giftemp/corner_d.jpg giftemp/corner_labels.jpg label_corners.gif
convert -quality 100 label_corners.gif \( -clone 0 -set delay 250 \) -swap 0 +delete \( +clone -set delay 250 \) +swap +delete label_corners.gif

rm -r giftemp

# Create score block finding animation
mkdir giftemp
cp table_ends.jpg table_middle.jpg table_middles_of_middle.jpg table_middles_of_middle_add.jpg table_score_box.jpg giftemp

convert -quality 100 -delay 140 -loop 0 table_ends.jpg giftemp/table_middle.jpg giftemp/table_middles_of_middle.jpg giftemp/table_middles_of_middle_add.jpg giftemp/table_score_box.jpg find_score_blocks.gif
convert -quality 100 find_score_blocks.gif \( -clone 0 -set delay 250 \) -swap 0 +delete \( +clone -set delay 250 \) +swap +delete find_score_blocks.gif

rm -r giftemp


# Create left score threshold animation
mkdir giftemp
cp left_score_blocks.jpg left_score_blocks_black_white.jpg giftemp

convert -quality 100 -delay 140 -loop 0 giftemp/left_score_blocks.jpg giftemp/left_score_blocks_black_white.jpg left_threshold.gif

rm -r giftemp

# Create right score threshold animation
mkdir giftemp
cp right_score_blocks.jpg right_score_blocks_black_white.jpg giftemp

convert -quality 100 -delay 140 -loop 0 giftemp/right_score_blocks.jpg giftemp/right_score_blocks_black_white.jpg right_threshold.gif

rm -r giftemp


# Create total animation
mkdir giftemp
cp *.jpg giftemp

convert -quality 100 -delay 40 -loop 0 giftemp/testdata.jpg giftemp/large.jpg giftemp/found_blue.jpg giftemp/found_corners.jpg giftemp/lower_long_side.jpg giftemp/large_straight.jpg giftemp/corner_a.jpg giftemp/corner_b.jpg giftemp/corner_c.jpg giftemp/corner_d.jpg giftemp/corner_labels.jpg giftemp/table_middle.jpg giftemp/table_middles_of_middle.jpg giftemp/table_middles_of_middle_add.jpg giftemp/table_score_box.jpg giftemp/left_score_blocks.jpg giftemp/left_score_blocks_black_white.jpg giftemp/centers_left.jpg giftemp/left_score.jpg giftemp/right_score_blocks.jpg giftemp/right_score_blocks_black_white.jpg giftemp/centers_right.jpg giftemp/right_score.jpg algorithm.gif

rm -r giftemp
