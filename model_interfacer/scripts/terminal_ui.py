#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import gensim
from sklearn.externals import joblib
import rospy
from std_msgs.msg import String
# from pprintpp import pprint
import json
import rospkg

PACKAGE_PATH = rospkg.RosPack().get_path("model_interfacer")
MODEL_FILEPATH = PACKAGE_PATH + "/scripts/models/"
import curses


import sys,os
import curses
#encoding: utf-8

"""
The application module supplies the Application object, which provides the core
utilities for running an application with gruepy.
"""
import locale

locale.setlocale(locale.LC_ALL, '')

import curses
from curses import wrapper
import time
import numpy as np
import pigpio

# import pdb


pi = pigpio.pi()


class PiGpioStatusWindow(object):
    """
    Shows the status of pigpio.
    """

    def __init__(self, screen):
        self.screen = screen
        screen_dimensions = screen.getmaxyx()
        height = int(screen_dimensions[0] / 5.0)
        width = screen_dimensions[1] - 2
        self.win = curses.newwin(height, width, 1, 1)
        self.dimensions = self.win.getmaxyx()

    def show_status(self, pigpio_status):
        self.win.clear()
        self.win.border(0)
        self.win.addnstr(0, self.dimensions[1] / 2 - 3, "PIGPIO STATUS", 13, curses.color_pair(2))
        text = "Connected: " + pigpio_status
        self.win.addnstr(2, 1, text, 14)

        # Hide cursor
        self.win.leaveok(1)

        # Update virtual window
        self.win.noutrefresh()


def setup_screen(screen):
    """
    Performs setup of the screen.
    """
    # Handle the keypad and special keys
    screen.keypad(True)

    # Set screen to getch() is non-blocking
    screen.nodelay(1)

    # define some colors
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(6, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

    # add border
    screen.border(0)


def destroy_screen(screen):
    """
    Destroy an ncurses screen, allowing a clean return to the calling terminal.
    """
    curses.nocbreak()
    screen.keypad(False)
    curses.echo()
    curses.endwin()


def main():
    """
    Provides an ncurses-based interface for control of the GPIO-related functionality of the AIY Voice Hat.
    """
    screen = curses.initscr()

    # Turn off character echoing
    curses.noecho()

    # Instantly react to key presses (ie: don't wait for the ENTER key to be pressed)
    curses.cbreak()

    curses.start_color()

    # Setup the screen
    setup_screen(screen)

    # Create Windows
    pigpiostatuswin = PiGpioStatusWindow(screen)

    while True:
        # draw pigpio status
        pigpiostatuswin.show_status(pi.connected)

        # Update physical screen
        curses.doupdate()

        # Check for keystrokes, and process
        keyb = screen.getch()

        # Sleep to get ~10Hz regresh rate
        time.sleep(0.1)


if __name__ == '__main__':
    try:
        screen = wrapper(main())
    except KeyboardInterrupt:
        destroy_screen(screen)
        pass