# Football score detector


![table](docs/table.jpg)



- labeling algorithm
- largest from labeling
- rotate the image. labeling does not work.. hmmm..

next try:

- color segmenting
- find contours from masked
- find corners.. how? using opencv feature detection or else.. or simply sorting pixels based on coordinates..
- find ends.. hmm how. two closest pairs from the corner coordinates because the table is not a square

- remove find contours from masked, otherwise same

That did not work if the table was too straight because upper most pixel

- Find

    rect = cv2.minAreaRect(non_zero_pixels)
    points = cv2.cv.BoxPoints(rect)
    points = np.int0(np.around(points))

  and remove own hacky find corners function.

- This leads to pretty solid table end detection and is fast because internal functions

- next challenge: how to get position of score dots?
 try: half of corner dots, and half again





# Install


    sudo apt-get install git python-opencv python-pip

