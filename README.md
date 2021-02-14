# OpenCV Document Scanner 

Document scanner written in Python utilizing its OpenCV2 library.  

Designed to create a clean greyscale scan of a document given picture taken at any angle (within reason).  The program works by applying a four point perspective transform based on the position of the corners of the page and its edge contours.  The edge contours are obtained using OpenCV2's findContours function.  For this function to work, canny edge detection needs to be used to draw the edges of the page.  All steps of the process are displayed during runtime. 

The program was designed to run in the command line via Python's argparse interface.

```bash
python image_transform.py -i "image filepath"
```

