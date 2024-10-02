"""This exlpains the coding with some examples.

It extends the "Coding rules" document and illustrates the points.

do: things that are fine to do.
don't: things that are bad. They may affect the evaluation negatively
always: do it like that, always. If you have a good reason to deviate, please ask the assistant. If not, it *may* reduces the points.
never: don't do this, never. If you find yourself compelled to do it, please discuss with the assistant. If not, it *may* reduces the points.
NEVER: this does not belong into a Python script and will cost points (or may even result in 0 points).
"""

# this import the plotting library
import matplotlib.pyplot as plt  # always

from numpy import *  # NEVER: do not use * to import something


# this is a comment  # do
# for multiple lines
# PyCharm can help you: mark everything, then press Ctrl + / (or check in the menu), this will comment/uncomment things
"""This is NOT a comment. It's a string
Do not use this!""" # don't

# plotting guide

plt.figure()  # always: whenever you create a new figure, create it explicitly. Otherwise, it may plots into other figures.
plt.plot(4, 3, label="first points")
plt.plot(10, 2, label="second points")
plt.xlabel("depth (m)")  # always: axes need labels and units! Otherwise the plot is meaningless
plt.ylabel("height (cm)")  # always
plt.title("Comparison of volumes")  # do: describe what is in the plot
plt.legend()  # do: this will plot the labels
plt.savefig("Comparison_of_volumes.pdf")  # always: save the figure and don't use spaces
plt.close()  # this closes the figure

plt.show()  # don't: you can use plt.show! But please uncomment before handing in
# plt.show()  # do
# always: place plt.show() *after* savefig


# TRICK: we can use a variable and the replace function to make the title creation and savefig work better together
# (rest of plot omitted here)
title = "Comparison of volumes"
plt.title(title)
plt.savefig(title.replace(" ", "_") + ".pdf")

age = 42  # do: write what it is!
a = 42  # don't (never): a is meaningless and makes reading the code very hard.
