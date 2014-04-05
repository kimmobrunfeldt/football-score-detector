# Football score detector

The program analyzes this picture

![table](docs/table.jpg)

and outputs following JSON

```json
{
  "leftScore": 9,
  "rightScore": 3
}
```

## Install

**Dependencies**
- OpenCV >= 2.4.4
- Numpy
- Scipy

These might be tricky to install. I created new Ubuntu 13.10 vagrant box and installed dependencies with apt-get:

    sudo apt-get install git python-opencv python-pip python-scipy



