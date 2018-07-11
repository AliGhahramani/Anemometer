import sys, random
import anemUI
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QVBoxLayout, QLabel, QPushButton, QListWidgetItem, \
    QHBoxLayout



class AnemometerSiteWidget(QWidget):
    def __init__(self, anemID="anemometerID", anemType="anemometerType", siteID="siteID", parent=None):
        super(AnemometerSiteWidget, self).__init__(parent)

        layout = QHBoxLayout()
        layout.addWidget(QLabel(anemID))
        layout.addWidget(QLabel(anemType))
        layout.addWidget(QLabel(siteID))
        layout.addWidget(QPushButton("A useless button"))

        self.setLayout(layout)

# Opens data stream and adds anemometers to UI list.
def _process_input():
    readings_generator = anemUI.read_input(True)
    for reading in readings_generator:
        if reading.anemometer_id not in anemometers:
            print("Found new anemometer: ", reading.anemometer_id)
            anemometer_type = "Duct"
            if reading.is_duct6:
                anemometer_type = "Duct6"
            elif reading.is_room:
                anemometer_type = "Room"

            # Some weird multithreading bug from the below code, so instead just using text in the list for now. 
            # item = QListWidgetItem(list)
            # item_widget = AnemometerSiteWidget(reading.anemometer_id, anemometer_type, reading.site)
            # item.setSizeHint(item_widget.sizeHint())
            # list.addItem(item)
            # list.setItemWidget(item, item_widget)
            list.addItem(reading.anemometer_id + "\t" + anemometer_type + "\t" + reading.site)

            anemometers.add(reading.anemometer_id)


if __name__ == '__main__':
    anemometers = set()

    # set up app
    app = QApplication(sys.argv)
    window = QWidget()
    window_layout = QVBoxLayout(window)
    title = QLabel("Connected anemometers")
    list = QListWidget()
    window_layout.addWidget(title)
    window_layout.addWidget(list)
    window.setLayout(window_layout)
    list.addItem("Anemometer ID\tType\tSite ID")

    # start thread for reading in input
    t = threading.Thread(target=_process_input)
    t.daemon = True
    t.start()

    # start the app
    window.show()
\
    sys.exit(app.exec_())
    
