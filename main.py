#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import SessionEntry
import RecordData

def main(argv):
    app = QtWidgets.QApplication(argv)
    sessionEntryWindow = SessionEntry.SessionEntryWindow()
    sessionEntryWindow.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main(sys.argv)
