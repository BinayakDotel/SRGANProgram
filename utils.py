# import the necessary packages
import config
from matplotlib.pyplot import subplots
from matplotlib.pyplot import savefig
from matplotlib.pyplot import title
from matplotlib.pyplot import xticks
from matplotlib.pyplot import yticks
from matplotlib.pyplot import show
from tensorflow.keras.preprocessing.image import array_to_img
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os

def zoom_into_images(image, imageTitle):
    # create a new figure with a default 111 subplot.
    (fig, ax) = subplots()
    im = ax.imshow(array_to_img(image[::-1]), origin="lower")
    title(imageTitle)
    
    # zoom-factor: 2.0, location: upper-left
    axins = zoomed_inset_axes(ax, 2, loc=2)
    axins.imshow(array_to_img(image[::-1]), origin="lower")
    
    # specify the limits.
    (x1, x2, y1, y2) = 20, 40, 20, 40
    
    # apply the x-limits.
    axins.set_xlim(x1, x2)
    
    # apply the y-limits.
    axins.set_ylim(y1, y2)
    
    # remove the xticks and yticks
    yticks(visible=False)
    xticks(visible=False)
    
    # make the line.
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
    
    # build the image path and save it to disk
    imagePath = os.path.join(config.BASE_IMAGE_PATH,
        f"{imageTitle}.png")
    savefig(imagePath)
    
    # show the image
    show()