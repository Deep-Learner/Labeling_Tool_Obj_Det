import sys
import os
import glob
import csv
import numpy as np
import cv2
import darkdetect

from PyQt5.QtWidgets import *
from PyQt5 import uic, QtCore

from PyQt5.QtGui import *
from PyQt5.QtCore import *



class Labeling_Tool_GUI(QMainWindow):

    def __init__(self, scale_factor, margin_factor=0.85):
        super(Labeling_Tool_GUI, self).__init__()
        # Set the window to the foreground
        # self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)

        # print("self.getWindowFlags() =", self.windowFlags())
        self.gui_path = os.getcwd() + os.sep
        self.parrent_path = (os.sep).join(os.getcwd().split(os.sep)[:-1]) + os.sep
        self.anno_path = self.parrent_path + "Annotation_Files" + os.sep
        self.pred_path = self.parrent_path + "Predicted_Labels" + os.sep
        self.video_path = self.parrent_path + "Videos" + os.sep + "ARG 19 SS16 OBC NEU POV" + os.sep
        print("self.gui_path =", self.gui_path)
        print("self.parrent_path =", self.parrent_path)
        print("self.anno_path =", self.anno_path)
        print("self.pred_path =", self.pred_path)
        print("self.video_path =", self.video_path)

        for cur_path in [self.gui_path, self.parrent_path, self.anno_path, self.pred_path, self.video_path]:
            if not os.path.isdir(cur_path):
                os.makedirs(cur_path)

        self.cur_video_path = ""

        if self.cur_video_path == "":
            self.all_videos_in_video_dir = [elem for elem in glob.glob(self.video_path + "*.mp4")]
        else:
            self.all_videos_in_video_dir = [self.cur_video_path]

        self.scale_factor = scale_factor
        self.scene_factor = margin_factor
        self.org_height, self.org_width, self.org_channels = 1080, 1920, 3
        if len(self.all_videos_in_video_dir) > 0:
            self.cur_video_path = self.all_videos_in_video_dir[0]

            # OpenCV stuff:
            # self.cur_video_path = "D:/Dropbox/Dropbox/DFKI (Arbeit)/WRC/Videos/ARG 19 SS16 OBC NEU POV/ARG 19 SS16 OBC NEU POV.mp4"
            temp_vidcap = cv2.VideoCapture(self.cur_video_path)
            success, image = temp_vidcap.read()
            self.total_num_frames = int(temp_vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            temp_vidcap.release()

            self.org_height, self.org_width, self.org_channels = image.shape
            # self.margin_right = 175
            # self.margin_bottom = 100
            # self.scale_height, self.scale_width, _ = (self.scale_factor * np.array(image.shape)).astype(int)
            self.scale_height, self.scale_width = int(self.scale_factor * self.org_height), int(self.scale_factor * self.org_width)
            # self.scene_height, self.scene_width = (0.85 * np.array([self.scale_height, self.scale_width])).astype(int)
            # self.scene_height, self.scene_width = int(self.scale_height - self.margin_bottom), int(self.scale_width - self.margin_right)
            self.scene_height, self.scene_width = int(self.scene_factor * self.scale_height), int(self.scene_factor * self.scale_width)
            self.margin_right = self.scale_width - self.scene_width
            self.margin_bottom = self.scale_height - self.scene_height
            print("self.org_height, self.org_width, self.org_channels =", self.org_height, self.org_width, self.org_channels)
            print("self.scale_height, self.scale_width =", self.scale_height, self.scale_width)
            print("self.scene_height, self.scene_width =", self.scene_height, self.scene_width)
        else:
            self.cur_video_path = ""
            self.total_num_frames = -1


            # self.margin_right = 175
            # self.margin_bottom = 100
            self.scale_height, self.scale_width, _ = (self.scale_factor * np.array([self.org_height, self.org_width, self.org_channels])).astype(int)
            # self.scene_height, self.scene_width = (0.85 * np.array([self.scale_height, self.scale_width])).astype(int)
            # self.scene_height, self.scene_width = int(self.scale_height - self.margin_bottom), int(self.scale_width - self.margin_right)
            self.scene_height, self.scene_width = int(self.scene_factor * self.scale_height), int(self.scene_factor * self.scale_width)
            self.margin_right = self.scale_width - self.scene_width
            self.margin_bottom = self.scale_height - self.scene_height
            print("self.org_height, self.org_width, self.org_channels =", self.org_height, self.org_width, self.org_channels)
            print("self.scale_height, self.scale_width =", self.scale_height, self.scale_width)
            print("self.scene_height, self.scene_width =", self.scene_height, self.scene_width)

        self.frame_counter = 0
        self.cur_class = 0
        self.class_list = ["No Class",
                           "Safe Person",
                           "Partially Safe Person",
                           "Unsafe Person"
                           ]

        self.class_colors = [(0, 163, 232),
                             (0, 200, 0),
                             (232, 163, 0),
                             (200, 0, 0)
                             ]

        """
        self.tag_list = sorted(["blurry",
                               "covered",
                               "noisy",
                               "size-L",
                               "size-M",
                               "size-S"
                               ])
        """

        self.config_path = os.getcwd() + os.sep + "config.csv"
        self.tag_list = []
        if os.path.isfile(self.config_path):
            config_file = open(self.config_path, 'r', encoding="utf-8")
            class_csv_Reader = list(csv.reader(config_file, delimiter=','))
            print("class_csv_Reader =", class_csv_Reader)
            self.tag_list = [tag for row in class_csv_Reader for tag in row[1:] if row[0] == "tags"]
            print("self.tag_list =", self.tag_list)
            config_file.close()

        if not os.path.isfile(self.config_path) or len(self.tag_list) == 0:
            config_file = open(self.config_path, 'a+')
            config_file.write("tags,blurry,covered,noisy\n")
            self.tag_list = ["blurry", "covered", "noisy"]
            config_file.close()

        self.tag_list = sorted(list(set(self.tag_list)))
        self.used_tags = []

        self.load_layout()

        self.graphics_scene.chk_box_list = self.chk_box_list
        self.graphics_scene.tag_list = self.tag_list
        self.graphics_scene.combo_box = self.combo_box

        # For zoom-function
        self._zoom = 0.0
        self.graphics_viewer.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.graphics_viewer.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # self.graphics_viewer.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        # self.graphics_viewer.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        # self.graphics_viewer.setTransformationAnchor(QGraphicsView.NoAnchor)
        # self.graphics_viewer.setResizeAnchor(QGraphicsView.NoAnchor)
        self.graphics_viewer.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphics_viewer.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # self.graphics_viewer.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # self.graphics_viewer.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self.m_originalX = 0.0
        self.m_originalY = 0.0
        self.m_moving = False
        # self.graphics_scene.graphics_view = self.graphics_viewer

        # self.setMinimumSize(self.scale_width, self.scale_height)
        # self.setMaximumSize(self.scale_width, self.scale_height)
        self.setFixedWidth(self.scale_width)
        self.setFixedHeight(self.scale_height)

        # self.grabKeyboard()
        # self.setMouseTracking(True)
        # self.installEventFilter(self)
        self.load_frame()

    def load_layout(self):
        uic.loadUi("My_Labeling_GUI.ui", self)  # Load the .ui file
        self.setGeometry(0, 0, self.scale_width, self.scale_height)

        self.main_window = self.findChild(QMainWindow, 'mainWindow')
        self.centralwidget = self.findChild(QWidget, 'centralwidget')

        # Find Menu-Bar Entries and set the corresponding methods: ---------------------------
        self.action_Load_Video = self.findChild(QAction, 'action_Load_Video')
        self.action_Set_Anno_Path = self.findChild(QAction, 'action_Set_Anno_Path')
        self.action_Set_Pred_Path = self.findChild(QAction, 'action_Set_Pred_Path')
        self.action_Load_Annotations = self.findChild(QAction, 'action_Load_Annotations')
        self.action_Load_Predictions = self.findChild(QAction, 'action_Load_Predictions')
        self.action_Add_new_Tag = self.findChild(QAction, 'action_Add_new_Tag')
        self.action_Change_Window_Size = self.findChild(QAction, 'action_Change_Window_Size')
        self.action_Change_Margin_Size = self.findChild(QAction, 'action_Change_Margin_Size')

        self.action_Load_Video.setShortcut('Ctrl+L')
        self.action_Load_Video.triggered.connect(self.load_video)


        self.action_Set_Anno_Path.triggered.connect(self.set_anno_path)
        self.action_Set_Pred_Path.triggered.connect(self.set_pred_path)

        self.action_Load_Annotations.setShortcut('Ctrl+G')
        self.action_Load_Annotations.triggered.connect(self.load_anno)

        self.action_Load_Predictions.setShortcut('Ctrl+P')
        self.action_Load_Predictions.triggered.connect(self.load_pred)


        self.action_Add_new_Tag.setShortcut('Ctrl+T')
        self.action_Add_new_Tag.triggered.connect(self.add_new_tag)

        self.action_Change_Window_Size.setShortcut('Ctrl+S')
        self.action_Change_Window_Size.triggered.connect(self.change_window_size)

        self.action_Change_Margin_Size.setShortcut('Ctrl+M')
        self.action_Change_Margin_Size.triggered.connect(self.change_margin_size)
        # End of Menu-Bar definition: --------------------------------------------------------


        # Find all labels in the layout: -----------------------------------------------------
        label_width = 70
        label_height = 50
        label_x_pos = 20
        label_y_pos = self.scale_height - self.margin_bottom + 15
        self.label_1 = self.findChild(QLabel, 'label_1')
        # self.label_1.setGeometry(QRect(label_x_pos, label_y_pos, label_width, label_height))
        if not self.video_path == "":
            self.label_1.setText("Frame:")
        self.label_1.setFixedWidth(label_width)
        self.label_1.setFixedHeight(label_height)
        # End of label definition ------------------------------------------------------------

        # Define LineEdit (Textline): --------------------------------------------------------
        edit_width = 100
        edit_height = 50
        edit_x_pos = 20
        edit_y_pos = self.scale_height - self.margin_bottom + 15
        self.text_edit = self.findChild(QLineEdit, 'lineEdit')
        # self.label_1.setGeometry(QRect(label_x_pos, label_y_pos, label_width, label_height))
        if not self.video_path == "":
            self.text_edit.setText(str(self.frame_counter))
        self.text_edit.setFixedWidth(edit_width)
        self.text_edit.setFixedHeight(edit_height)
        # self.text_edit.cursorPositionChanged.connect(self.text_edit_get_Focus)
        self.text_edit.returnPressed.connect(self.text_edit_enter)
        # self.text_edit.textChanged.connect(self.text_edit_textChanged)
        # End of LineEdit definition ---------------------------------------------------------


        # Find all sliders in the layout: ----------------------------------------------------
        slider_width = 300
        slider_height = 35
        slider_x_pos = label_x_pos + label_width + 10
        slider_y_pos = label_y_pos + slider_height/4# self.scale_height - self.margin_bottom + slider_height/2 + 5
        self.slider_frames = self.findChild(QSlider, 'frame_slider')
        if self.cur_video_path == "":
            self.slider_frames.setMaximum(0)
        else:
            self.slider_frames.setMaximum(self.total_num_frames - 1)
        # self.slider_frames.setGeometry(QRect(slider_x_pos, slider_y_pos, slider_width, slider_height))
        self.slider_frames.setFixedWidth(slider_width)
        self.slider_frames.setFixedHeight(slider_height)
        # End of sliders definition ----------------------------------------------------------

        # Find all buttons in the layout: ----------------------------------------------------
        save_btn_width = self.scale_width - (self.scene_width + 30)  # 125
        save_btn_height = 50
        save_btn_x = self.scale_width - save_btn_width - 10
        save_btn_y = label_y_pos
        self.btn_save_frame = self.findChild(QPushButton, 'pushButton')
        # self.btn_save_frame.setGeometry(QRect(save_btn_x, save_btn_y, save_btn_width, save_btn_height))
        self.btn_save_frame.setFixedWidth(save_btn_width)
        self.btn_save_frame.setFixedHeight(save_btn_height)


        forw_btn_width = 30
        forw_btn_height = save_btn_height
        forw_btn_x = slider_x_pos + slider_width + 10
        forw_btn_y = save_btn_y
        self.btn_frame_back = self.findChild(QPushButton, 'pushButton_2')
        # self.btn_frame_back.setGeometry(QRect(forw_btn_x, forw_btn_y, forw_btn_width, forw_btn_height))
        self.btn_frame_back.setFixedWidth(forw_btn_width)
        self.btn_frame_back.setFixedHeight(save_btn_height)


        self.btn_frame_forw = self.findChild(QPushButton, 'pushButton_3')
        # self.btn_frame_forw.setGeometry(QRect(forw_btn_x+forw_btn_width+5, forw_btn_y, forw_btn_width, forw_btn_height))
        self.btn_frame_forw.setFixedWidth(forw_btn_width)
        self.btn_frame_forw.setFixedHeight(save_btn_height)
        # End of button definition -----------------------------------------------------------

        # Define toggle-Button to activate tagging: ------------------------------------------
        toggle_button_offset_width = 25
        toggle_button_w = self.scale_width - self.scene_width - 40
        toggle_button_h = 35
        toggle_button_x = self.scene_width + toggle_button_offset_width  # checkbox_w/2
        toggle_button_y = toggle_button_h
        self.toggle_button = QPushButton("Tagging off", self)
        # self.toggle_button.setGeometry(toggle_button_x, toggle_button_y, toggle_button_w, toggle_button_h)
        # In the beginning there is no person, so we do not allow to activate the tagging mode.
        self.toggle_button.setCheckable(True)

        self.toggle_button.clicked.connect(self.toggle_button_method)
        # End of toggle-Button definition ----------------------------------------------------


        # Define all checkboxes through tag_list: --------------------------------------------
        self.chk_box_list = []

        # checkbox_size = (80, 20)
        checkbox_w = 80
        checkbox_h = 20
        checkbox_offset_width = 25
        checkbox_offset_height = 10
        last_checkbox_coordinates = None
        for i, elem in enumerate(self.tag_list):
            self.chk_box_list.append(QCheckBox(self.centralwidget))
            self.chk_box_list[i].setObjectName(u"checkBox_" + str((i+1)))
            checkbox_x = self.scene_width + checkbox_offset_width # checkbox_w/2
            checkbox_y = (i+1) * (checkbox_h + checkbox_offset_height) - checkbox_offset_height + toggle_button_y
            # self.chk_box_list[i].setGeometry(QRect(checkbox_x, checkbox_y, checkbox_w, checkbox_h))
            last_checkbox_coordinates = (checkbox_x, checkbox_y, checkbox_w, checkbox_h)
            self.chk_box_list[i].setText(self.tag_list[i])

            # In the beginning there is no person, so we do not allow the tagging mode.
            self.chk_box_list[i].setChecked(False)
            self.chk_box_list[i].setCheckable(False)
            self.chk_box_list[i].setStyleSheet("color : rgb(150,150,150)")

            # Set the function for all checkboxes.
            self.chk_box_list[i].stateChanged.connect(self.checkbox_method)
        # End of checkbox definition ---------------------------------------------------------

        # Define Combo Box for classes: ------------------------------------------------------
        self.combo_box = QComboBox(self.centralwidget)
        combo_box_width_offset = 40
        # self.combo_box.setAlignment(Qt.AlignLeft)
        combo_box_x = self.scene_width + combo_box_width_offset/2
        combo_box_y = last_checkbox_coordinates[1] + checkbox_offset_height + checkbox_h
        combo_box_w = self.scale_width - self.scene_width - combo_box_width_offset
        combo_box_h = 25
        print("combo_box_x, combo_box_y, combo_box_w, combo_box_h =", combo_box_x, combo_box_y, combo_box_w, combo_box_h)
        # self.combo_box.setGeometry(QRect(combo_box_x, combo_box_y, combo_box_w, combo_box_h))
        # self.combo_box.setStyleSheet('selection-background-color: rgb(0,0,0,0)')
        # self.combo_box.setStyleSheet('selection-color: rgb(200,200,200,50)')

        # colors = ["blue", "green", "orange", "red"]
        alpha = 128
        colors = [QColor(0, 163, 232, alpha), QColor(0, 163, 0, alpha), QColor(232, 163, 0, alpha), QColor(255, 0, 0, alpha)]

        for i, elem in enumerate(self.class_list):
            # print("self.class_list[i] =", self.class_list[i])
            # self.combo_box.setItemText(i, self.class_list[i])
            # cur_color = QColor(colors[i])
            cur_color_QT = QColor(colors[i])
            # pixmap = QPixmap(100, 100)
            # pixmap.fill(cur_color)

            # self.combo_box.addItem(self.class_list[i])
            self.combo_box.addItem(self.class_list[i])
            self.combo_box.setItemData(i, cur_color_QT, Qt.BackgroundRole)

            # self.combo_box.setItemData(i, cur_color_QT, Qt.SelectionBackgroundRole)
            # self.combo_box.addItem(QIcon(pixmap), self.class_list[i])

            # pal = self.combo_box.palette()
            # pal.setColor(QPalette.Button, cur_color)
            # self.cb.setPalette(pal)
        self.combo_box.activated.connect(self.combo_box_method)
        # End of Combo Boc definition --------------------------------------------------------


        # Define toggle-Button for reclassification: -----------------------------------------
        reset_button_offset_height = 35
        reset_button_w = toggle_button_w
        reset_button_h = toggle_button_h
        reset_button_x = toggle_button_x
        reset_button_y = combo_box_y + combo_box_h + reset_button_offset_height
        self.reset_button = QPushButton("Reset", self)
        # self.reset_button.setGeometry(reset_button_x, reset_button_y, reset_button_w, reset_button_h)
        # In the beginning there is no person, so we do not allow to activate the tagging mode.
        # self.reset_button.setCheckable(False)

        self.reset_button.clicked.connect(self.reset_button_method)
        # End of toggle-Button definition (for reclassification) -----------------------------

        # Define a drag-vs-draw toggle-Button: -----------------------------------------------
        drag_draw_button_offset_height = 35
        drag_draw_button_w = reset_button_w
        drag_draw_button_h = reset_button_h
        drag_draw_button_x = reset_button_x
        drag_draw_button_y = reset_button_y + reset_button_h + drag_draw_button_offset_height
        self.drag_draw_button = QPushButton("Draw Mode", self)
        # self.drag_draw_button.setGeometry(drag_draw_button_x, drag_draw_button_y, drag_draw_button_w, drag_draw_button_h)
        self.drag_draw_button.setCheckable(True)

        self.drag_draw_button.clicked.connect(self.drag_draw_button_method)
        # End of drag-vs-draw toggle-Button definition (for reclassification) ----------------


        # Find the QGraphicsView in the layout and add underlying QGraphicsScene: ------------
        """
        self.graphics_viewer = self.findChild(QGraphicsView, 'graphicsView')
        self.graphics_scene = GraphicsScene(self.scene_width, self.scene_height)
        # self.graphics_scene = GraphicsScene(self)
        """
        self.graphics_scene = GraphicsScene(self.scene_width, self.scene_height)
        # self.graphics_viewer = self.findChild(QGraphicsView, 'graphicsView')
        self.graphics_viewer = GraphicsView(self.centralwidget)
        # self.graphics_viewer.setGeometry(QRect(9, 9, 961, 681))
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(1)
        sizePolicy1.setHeightForWidth(self.graphics_viewer.sizePolicy().hasHeightForWidth())
        self.graphics_viewer.setSizePolicy(sizePolicy1)
        self.graphics_viewer.setSizePolicy(sizePolicy1)
        self.graphics_viewer.setMouseTracking(True)
        self.graphics_viewer.setTabletTracking(True)
        self.graphics_viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_viewer.drag_draw_button = self.drag_draw_button

        self.graphics_viewer.setScene(self.graphics_scene)
        self.graphics_viewer.setFixedWidth(self.scene_width)
        self.graphics_viewer.setFixedHeight(self.scene_height)

        self.graphics_viewer.my_scene = self
        self.graphics_scene.main_window = self
        print("self.graphics_viewer.size() =", self.graphics_viewer.size())
        # End of the QGraphicsView and QGraphicsScene definition -----------------------------


        # Define Layouts ---------------------------------------------------------------------
        self.outher_vertical_layout = QVBoxLayout()


        self.horizontal_layout = QHBoxLayout()
        # self.horizontal_layout.addWidget(self.graphics_scene)
        self.horizontal_layout.addWidget(self.graphics_viewer)
        # self.horizontal_layout.addStretch(1)

        self.inner_vertical_layout = QVBoxLayout()
        # self.vertical_layout.setGeometry(QRect(self.scene_width, 0, self.scale_width, self.scene_height))
        self.inner_vertical_layout.addWidget(self.toggle_button)

        for chk_box in self.chk_box_list:
            self.inner_vertical_layout.addWidget(chk_box)

        self.inner_vertical_layout.addWidget(self.combo_box)
        self.inner_vertical_layout.addWidget(self.reset_button)
        self.inner_vertical_layout.addWidget(self.drag_draw_button)
        self.inner_vertical_layout.addStretch(1)

        # self.horizontal_layout.setGeometry(QRect(0, 0, self.scale_width, self.scene_height))
        # vertical_layout = QHBoxLayout()
        # self.grid_layout = self.findChild(QGridLayout, 'gridLayout')
        # self.gridLayoutWidget.setGeometry(QRect(0, 0, self.scale_width, self.scale_height))

        self.inner_horizontal_layout = QHBoxLayout()
        self.inner_horizontal_layout.addWidget(self.label_1)
        self.inner_horizontal_layout.addWidget(self.text_edit)
        self.inner_horizontal_layout.addWidget(self.slider_frames)
        self.inner_horizontal_layout.addWidget(self.btn_frame_back)
        self.inner_horizontal_layout.addWidget(self.btn_frame_forw)
        self.inner_horizontal_layout.addStretch(1)
        self.inner_horizontal_layout.addWidget(self.btn_save_frame)

        self.horizontal_layout.addLayout(self.inner_vertical_layout)
        self.outher_vertical_layout.addLayout(self.horizontal_layout)
        self.outher_vertical_layout.addLayout(self.inner_horizontal_layout)
        # self.outher_vertical_layout.addStretch(1)



        self.centralwidget.setLayout(self.outher_vertical_layout)

        # End of Layout definition -----------------------------------------------------------



        # Bind the event handler on the buttons, sliders, checkboxes etc.
        self.btn_save_frame.clicked.connect(self.btn_save_frame_method)
        self.btn_frame_forw.clicked.connect(self.btn_forw_frame_method)
        self.btn_frame_back.clicked.connect(self.btn_back_frame_method)
        self.slider_frames.valueChanged.connect(self.slider_frames_method)

        self.update()
        self.show()

    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == Qt.Key_W:
                self.graphics_scene.w_key_is_pressed = True
            elif event.key() == Qt.Key_H:
                self.graphics_scene.h_key_is_pressed = True
            elif event.key() == Qt.Key_Z:
                self.graphics_scene.strg_key_is_pressed = True
            elif event.key() == Qt.Key_M:
                self.graphics_scene.m_key_is_pressed = True


        print("keyPressEvent (0) -> self.main_window.text_edit.hasFocus() =", event.key())
        print("keyPressEvent (01) -> self.main_window.text_edit.hasFocus() =", event.modifiers() == Qt.CTRL)
        print("keyPressEvent (1) -> self.main_window.text_edit.hasFocus() =", self.text_edit.hasFocus())

        if self.text_edit.hasFocus():
            cursor_pos = self.text_edit.cursorPosition()
            print("cursor_pos =", cursor_pos)
            old_text = self.text_edit.text()
            print("old_text =", old_text)
            if Qt.Key_0 <= event.key() <= Qt.Key_9:
                print("keyPressEvent (2) -> self.main_window.text_edit.hasFocus() =", self.text_edit.hasFocus())
                new_text = old_text[:cursor_pos] + chr(event.key()) + old_text[cursor_pos:]
                self.text_edit.setText(new_text)
                self.text_edit.setCursorPosition(cursor_pos+1)
                print("new_text =", new_text)
                # self.text_edit.setText(event.key().)
            elif event.key() == 16777234:
                self.text_edit.setCursorPosition(cursor_pos-1)
            elif event.key() == 16777236:
                self.text_edit.setCursorPosition(cursor_pos+1)
            elif event.modifiers() == 16777249 and event.key() == Qt.Key_A:
                self.text_edit.setSelection(0,len(old_text))


        # print("self.m_key_is_pressed =", self.m_key_is_pressed)
        # print("self.w_key_is_pressed =", self.w_key_is_pressed)
        # print("self.h_key_is_pressed =", self.h_key_is_pressed)

        super(Labeling_Tool_GUI, self).keyPressEvent(event)


    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == Qt.Key_W:
                self.graphics_scene.w_key_is_pressed = False
            elif event.key() == Qt.Key_H:
                self.graphics_scene.h_key_is_pressed = False
            elif event.key() == Qt.Key_M:
                self.graphics_scene.m_key_is_pressed = False
            # print("self.m_key_is_pressed =", self.m_key_is_pressed)


        # print("self.w_key_is_pressed =", self.w_key_is_pressed)
        # print("self.h_key_is_pressed =", self.h_key_is_pressed)
        super(Labeling_Tool_GUI, self).keyReleaseEvent(event)


    def text_edit_enter(self):
        print("text_edit_method -> self.text_edit.getText() =", self.text_edit.text())
        try:
            self.frame_counter = int(self.text_edit.text())
            self.load_frame()
            self.slider_frames.setValue(self.frame_counter)
        except:
            self.text_edit.setText(str(self.frame_counter))





        # self.frame_counter =

    """
    def text_edit_get_Focus(self):
        print("text_edit_get_Focus -Y self.text_edit.hasFocus() =", self.text_edit.hasFocus())
        if self.text_edit.hasFocus():
            self.text_edit.grabKeyboard()
        else:
            self.graphics_viewer.grabKeyboard()
        
    def text_edit_textChanged(self):
        print("text_edit_textChanged -> self.text_edit.hasFocus() =", self.text_edit.hasFocus())
    """

    def checkbox_method(self):
        if self.toggle_button.isChecked() and not self.graphics_scene.highlighted_person_id_in_coord_list is None:
            s = set()
            for elem in self.chk_box_list:
                if elem.isChecked():
                    s.add(elem.text())
            cur_coord = self.graphics_scene.coord_list[self.graphics_scene.highlighted_person_id_in_coord_list]
            print("checkbox_method -> cur_coord = ", self.graphics_scene.coord_list[self.graphics_scene.highlighted_person_id_in_coord_list])
            if len(cur_coord) == 2:
                self.graphics_scene.coord_list[self.graphics_scene.highlighted_person_id_in_coord_list].append(list(s))
            else:
                self.graphics_scene.coord_list[self.graphics_scene.highlighted_person_id_in_coord_list][-1] = list(s)
            print("self.graphics_scene.coord_list[self.graphics_scene.highlighted_person_id_in_coord_list] =", self.graphics_scene.coord_list[self.graphics_scene.highlighted_person_id_in_coord_list])

    def untoggle_tagging_button(self):


        self.graphics_scene.is_classifing = True
        self.graphics_scene.highlighted_person_id_in_coord_list = None
        self.graphics_scene.repaint()

        # set background color back to light-grey
        self.toggle_button.setStyleSheet("background-color : lightgrey")
        self.toggle_button.setText("Tagging off")
        self.toggle_button.setChecked(False)

        for chk_box in self.chk_box_list:
            chk_box.setChecked(False)
            chk_box.setCheckable(False)
            chk_box.setStyleSheet("color : rgb(150,150,150)")

    def toggle_button_method(self):
        """
        Idea how to tag persons:
            First, enable the tagging mode
                => all checkboxes get unchecked

            Second, click on a person.
                => This person gets highlighted by a thicker bounding box.
                   If this person was already tagged, those tags should be loaded

            Third, choose some tags.
                => The tags which are chosen, correspond to the currently highlighted person

            Fourth, either choose a new person or disable the tagging mode.
                => In both cases the last highlighted person get the previously chosen tags.
        :return:
        """


        if len(self.graphics_scene.coord_list) > 0:
            # self.toggle_button.setChecked(True)
            self.toggle_button.setCheckable(True)
        else:
            # self.toggle_button.setChecked(False)
            self.toggle_button.setCheckable(False)

        if len(self.graphics_scene.coord_list) > 0:
            # if button is checked
            if self.toggle_button.isChecked():
                self.graphics_scene.is_classifing = False

                # setting background color to light-green
                self.toggle_button.setStyleSheet("background-color : lightgreen")
                self.toggle_button.setText("Tagging on")

                # If Tagging Mode is on, then uncheck all pervious checkboxes:
                for chk_box in self.chk_box_list:
                    chk_box.setChecked(False)
                    chk_box.setCheckable(True)
                    if darkdetect.isLight() is None or darkdetect.isLight():
                        chk_box.setStyleSheet("color : rgb(50,50,50)")
                    else:
                        chk_box.setStyleSheet("color : rgb(255,255,200)")

            else:
                self.untoggle_tagging_button()
        else:
            self.untoggle_tagging_button()


    def reset_button_method(self):

        """
        Delete all bounding boxes and reset all button states (e.g. since not bounding box, therefore tagging
        toggle-button cannot be activated)
        """
        print("You called the toggle_button_recls_method() method.")

        # self.toggle_button.setCheckable(True)

        self.graphics_scene.is_classifing = True
        self.graphics_scene.temp_coord = []
        self.graphics_scene.coord_list = []
        self.graphics_scene.highlighted_person_id_in_coord_list = None
        # self.graphics_scene.cur_class = 0

        self.graphics_scene.chk_box_list = self.chk_box_list
        self.graphics_scene.tag_list = self.tag_list
        self.graphics_scene.combo_box = self.combo_box

        # self.toggle_button.setChecked(False)
        self.toggle_button_method()

        self.graphics_scene.repaint()

    def drag_draw_button_method(self):
        if self.drag_draw_button.isChecked():
            self.graphics_scene.drag_on = True

            # setting background color to light-green
            self.drag_draw_button.setStyleSheet("background-color : lightgreen")
            self.drag_draw_button.setText("Drag Mode")

            self.drag_draw_button.setChecked(True)

            self.graphics_viewer.setDragMode(QGraphicsView.ScrollHandDrag)
        else:
            self.graphics_scene.drag_on = False

            self.drag_draw_button.setStyleSheet("background-color : lightgrey")
            self.drag_draw_button.setText("Draw Mode")
            self.drag_draw_button.setChecked(False)

            self.graphics_viewer.setDragMode(QGraphicsView.NoDrag)

    def load_video(self):
        """
        ToDo: Use a QFileDialog to choose a path
        :return:
        """
        options = QFileDialog.Options()
        print(type(options))
        # .setGeometry(QRect(500, 500, 400, 200))
        # options |= QFileDialog.setGeometry(QRect(500, 500, 400, 200))
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, caption="Choose a mp4 Video",
                                                  directory=self.parrent_path,
                                                  filter="Video Files (*.mp4)",
                                                  options=options)

        if fileName:
            fileName = fileName.replace('/', os.sep)
            print("fileName =", fileName)
            self.cur_video_path = fileName
            self.video_path = (os.sep).join(fileName.split(os.sep)[:-1]) + os.sep
            temp_vidcap = cv2.VideoCapture(self.cur_video_path)
            print("self.cur_video_path =", self.cur_video_path)
            success, image = temp_vidcap.read()
            print("self.total_num_frames (old) =", self.total_num_frames)
            self.total_num_frames = int(temp_vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("self.total_num_frames (new) =", self.total_num_frames)
            temp_vidcap.release()

            self.slider_frames.setMaximum(self.total_num_frames - 1)
            self.text_edit.setText(str(self.frame_counter))

            # self.load_layout()

            # self.setFixedWidth(self.scale_width)
            # self.setFixedHeight(self.scale_height)

            # self.setMouseTracking(True)

            # self.frame_counter = 0

            self.load_frame()



    def set_anno_path(self):
        """
        ToDo: Use a QFileDialog to choose a path
        :return:
        """
        print("You called the set_anno_path() method.")

    def set_pred_path(self):
        """
        ToDo: Use a QFileDialog to choose a path
        :return:
        """
        print("You called the set_pred_path() method.")


    def centered_and_rel_to_xyxy(self, coord):
        x_center_abs = float(coord[1]) * float(self.scene_width)
        y_center_abs = float(coord[2]) * float(self.scene_height)
        width_abs = float(coord[3]) * float(self.scene_width)
        height_abs = float(coord[4]) * float(self.scene_height)

        x_left = x_center_abs - width_abs/2.0
        y_top = y_center_abs - height_abs/2.0
        x_right = x_center_abs + width_abs/2.0
        y_bottom = y_center_abs + height_abs/2.0
        return x_left, y_top, x_right, y_bottom



    def load_anno_or_preds(self, load_mode=0):
        """

        :param load_mode: There are three load_modes 0,1 or 2.
                          0 -> load predictions
                          1 -> load annotations
                          2 -> load annotations if available otherwise try to load predictions (if available)
        :return:
        """
        # pred_filename = self.pred_path + self.video_path.split(os.sep)[-2] + os.sep + self.video_path.split(os.sep)[-2] + "_" + str(self.frame_counter) + ".txt"
        # anno_filename = self.anno_path + self.video_path.split(os.sep)[-2] + os.sep + self.video_path.split(os.sep)[-2] + "_" + str(self.frame_counter) + ".txt"

        print("self.video_path =", self.video_path)
        print("self.cur_video_path =", self.cur_video_path)
        video_name = self.cur_video_path.split(os.sep)[-1].split('.')[0]
        pred_filename = self.pred_path + video_name + os.sep + video_name + "_" + str(self.frame_counter) + ".txt"
        anno_filename = self.anno_path + video_name + os.sep + video_name\
                        + "_" + str(self.frame_counter) + ".txt"

        if load_mode == 0:
            used_filename = pred_filename
            print("pred_filename =", used_filename)
        elif load_mode == 1:
            used_filename = anno_filename
            print("anno_filename =", used_filename)
        else:
            if os.path.isfile(anno_filename):
                used_filename = anno_filename
            elif os.path.isfile(pred_filename):
                used_filename = pred_filename
            else:
                # There are no ground truth data and no predictions for the current frame
                return None

        if os.path.isfile(used_filename):
            self.graphics_scene.reset_values()

            # ToDo: Load Annotations
            file = open(used_filename, 'r')
            data_str = list(csv.reader(file, delimiter=' '))
            file.close()

            for elem in data_str:
                x_left, y_top, x_right, y_bottom = self.centered_and_rel_to_xyxy(elem)
                if len(elem) == 5:
                    self.graphics_scene.coord_list.append([int(elem[0]), [(float(x_left), float(y_top)), (float(x_right), float(y_bottom))]])
                elif len(elem) > 5:
                    print("load Annotation elem[5:] =", elem[5])
                    print("elem[5:].split(',') =", elem[5].split(','))
                    saved_tag_ids = [saved_tag_str for saved_tag_str in elem[5].split(',') if saved_tag_str in self.tag_list]
                    # refPt.append([int(elem[0]), [(float(elem[1]), float(elem[2])), (float(elem[3]), float(elem[4]))], elem[5].split(',')])
                    self.graphics_scene.coord_list.append([int(elem[0]), [(float(x_left), float(y_top)), (float(x_right), float(y_bottom))], saved_tag_ids])
            self.graphics_scene.repaint()
            print("self.graphics_scene.coord_list =", self.graphics_scene.coord_list)
        else:
            # ToDo: Open a new small window with the message, that there are no annotations for this frame
            print("\n\n\n\t----------------------------------------")
            print("\tThere are no annotations for this frame.")
            print("\t----------------------------------------\n\n\n")


    def load_anno(self):
        """
        ToDo: Load the annotations from the annotation path
        :return:
        """
        print("You called the set_anno_path() method.")
        self.load_anno_or_preds(load_mode=1)




    def load_pred(self):
        """
        ToDo: Load the predictions from the prediction path
        :return:
        """
        self.load_anno_or_preds(load_mode=0)

    def add_new_tag(self):
        """
        ToDo: Open a new small window with a QEditLine in it, where a new Tag can be added.
        :return:
        """
        print("You called the add_new_tag() method.")

        # self.tag_list
        self.secondary_window = Secondary_Window(main_labeling_tool=self, mode="add_new_tag")

    def change_window_size(self):
        """
        ToDo: Open a new small window with a QEditLine in it, where a number (between 0.6 and 1) can be written.
              This number corresponds to scale_factor.
        :return:
        """
        self.secondary_window = Secondary_Window(main_labeling_tool=self, mode="change_window_size")
        print("You called the change_window_size() method.")

    def change_margin_size(self):
        self.secondary_window = Secondary_Window(main_labeling_tool=self, mode="change_margin_size")
        print("You called the change_margin_size() method.")

    def load_frame(self):
        if not self.cur_video_path == "":
            print("load_frame self.cur_video_path =", self.cur_video_path)
            vidcap = cv2.VideoCapture(self.cur_video_path)
            print("self.filename =", self.cur_video_path)
            if self.total_num_frames == -1:
                self.total_num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vidcap.set(1, self.frame_counter)  # The first argument (1) corresponds to cv2.CV_CAP_PROP_POS_FRAMES
            success, image = vidcap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (int(self.graphics_scene.width()+1), int(self.graphics_scene.height()+2)), interpolation=cv2.INTER_AREA)  # _LINEAR fastest
            vidcap.release()

            # self.graphics_scene.clear()
            h, w, c = image.shape
            print("w, h, c =", w, h, c)

            bytesPerLine = c * w
            convertToQtFormat = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)

            # p = convertToQtFormat.scaled(self.scene_width+5, self.scene_height+5)
            # p = convertToQtFormat.scaled(self.graphics_scene.scene_width, self.graphics_scene.scene_height)
            print("self.scene_width, self.scene_height =", self.scene_width, self.scene_height)

            # pixMap = QPixmap.fromImage(p)
            pixMap = QPixmap.fromImage(convertToQtFormat)
            self.graphics_scene.addPixmap(pixMap)
            self.graphics_scene.reset_values()
            self.graphics_scene.repaint()
            self.graphics_scene.update()


    def read_checkboxes(self):
        used_tags = []
        for idx, checkbox in enumerate(self.chk_box_list):
            if checkbox.isChecked():
                used_tags.append(self.tag_list[idx])
        return used_tags


    def btn_save_frame_method(self):
        # This is executed when the button is pressed
        if not self.cur_video_path == "":
            save_list = self.graphics_scene.coord_list.copy()

            cur_video_name = '.'.join(self.cur_video_path.split(os.sep)[-1].split('.')[:-1])
            cur_filename = self.anno_path + cur_video_name + os.sep + cur_video_name + "_" + str(self.frame_counter) + ".txt"
            print("cur_filename =", cur_filename)

            if not os.path.isdir(self.anno_path + cur_video_name):
                os.makedirs(self.anno_path + cur_video_name)

            if os.path.isfile(cur_filename):
                os.remove(cur_filename)

            file = open(cur_filename, 'a+')

            for elem in save_list:
                if len(elem) == 2:
                    cls, coord = elem
                else:
                    cls, coord, tags = elem

                print("self.scene_width =", self.scene_width)
                print("self.scene_height =", self.scene_height)
                # Compute some helper values
                x_left = float(min(coord[0][0], coord[1][0]))
                x_right = float(max(coord[0][0], coord[1][0]))
                y_top = float(min(coord[0][1], coord[1][1]))
                y_bottom = float(max(coord[0][1], coord[1][1]))

                # Compute the absolute values of x,y,w and h
                x_center_abs = (x_left + (x_right - x_left)/2.0)
                y_center_abs = (y_top + (y_bottom - y_top)/2.0)
                width_abs = x_right - x_left
                height_abs = y_bottom - y_top

                x_center_rel = x_center_abs/float(self.graphics_scene.width())
                y_center_rel = y_center_abs/float(self.graphics_scene.height())
                width_rel = width_abs/float(self.graphics_scene.width())
                height_rel = height_abs/float(self.graphics_scene.height())

                if len(elem) == 2:
                    save_str = str(cls) + " " + str(x_center_rel) + " " + str(y_center_rel) + " " + str(width_rel) + " " + str(height_rel) + "\n"
                else:
                    save_str = str(cls) + " " + str(x_center_rel) + " " + str(y_center_rel) + " " + str(width_rel) + " " + str(height_rel) + " " + ','.join(tags) +  "\n"

                print("save_str =", save_str)
                file.write(save_str)
            print("save_list =", save_list)
            file.close()
            self.used_tags = []
            self.toggle_button.setChecked(False)
            self.toggle_button_method()

            self.graphics_scene.reset_values()

            # self.graphics_scene.coord_list = []
            # self.graphics_scene.highlighted_person_id_in_coord_list = None

            self.frame_counter += 1
            self.text_edit.setText(str(self.frame_counter))
            self.slider_frames.setValue(self.frame_counter)
            self.load_frame()


    def btn_forw_frame_method(self):
        if not self.cur_video_path == "":
            save_list = self.graphics_scene.coord_list.copy()

            # self.graphics_scene.coord_list = []
            self.graphics_scene.reset_values()

            print("save_list =", save_list)
            self.frame_counter += 1
            self.text_edit.setText(str(self.frame_counter))
            self.slider_frames.setValue(self.frame_counter)
            self.untoggle_tagging_button()
            self.load_frame()

    def btn_back_frame_method(self):
        if not self.cur_video_path == "":
            save_list = self.graphics_scene.coord_list.copy()

            # self.graphics_scene.coord_list = []
            self.graphics_scene.reset_values()

            print("save_list =", save_list)
            self.frame_counter -= 1
            self.text_edit.setText(str(self.frame_counter))
            self.slider_frames.setValue(self.frame_counter)
            self.untoggle_tagging_button()
            self.load_frame()

    def slider_frames_method(self):
        # This is executed when the button is pressed
        if not self.cur_video_path == "":
            self.frame_counter = self.slider_frames.value()
            self.text_edit.setText(str(self.frame_counter))
            self.load_frame()

    def combo_box_method(self, idx):
        print("You changed the class to index ", idx, "\t\tThis cooresponds to the class name ", self.class_list[idx])
        self.cur_class = idx

        # self.graphics_scene.cur_class_GS = self.cur_class
        # self.graphics_scene.class_list_GS = self.class_list
        # self.graphics_scene.class_colors_GS = self.class_colors
        # self.graphics_scene.cur_class_color = self.class_colors[self.cur_class]
        self.graphics_scene.cur_class = self.cur_class

        if not self.graphics_scene.highlighted_person_id_in_coord_list is None:
            print("combo --- 0")
            self.graphics_scene.coord_list[self.graphics_scene.highlighted_person_id_in_coord_list][0] = idx
            print("combo --- 1")
            self.graphics_scene.repaint()
    """
    def fitInView(self, scale=True):
        unity = self.graphics_viewer.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
        self.graphics_viewer.scale(1 / unity.width(), 1 / unity.height())
        viewrect = self.graphics_viewer.viewport().rect()
        scenerect = self.graphics_viewer.transform().mapRect(self.graphics_viewer.rect())
        factor = min(viewrect.width() / scenerect.width(),
                     viewrect.height() / scenerect.height())
        self.graphics_viewer.scale(factor, factor)


    
    def wheelEvent(self, event):
        if self.graphics_scene.w_key_is_pressed or self.graphics_scene.h_key_is_pressed:
            pass
        else:
            # self.graphics_viewer.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            # self.graphics_viewer.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
            print("event.angleDelta().y() =", event.angleDelta().y())
            if event.angleDelta().y() > 0:
                factor = 1.1
                self._zoom += 0.1
            else:
                factor = 0.9
                self._zoom -= 0.1
            self._zoom = round(self._zoom, 1)
            print("self._zoom =", self._zoom)
            if self._zoom > 0:
                print("self._zoom > 0 -> True")
                print("factor =", factor)
                self.graphics_viewer.scale(factor, factor)
                # self.graphics_viewer.translate(0, -event.angleDelta().y() / 120.0)
                self.graphics_viewer.viewport()
            else:
                self.fitInView()
                self._zoom = 0.0
                self.graphics_scene.repaint()

            if self.drag_draw_button.isChecked():
                self.graphics_viewer.setDragMode(QGraphicsView.ScrollHandDrag)
            else:
                self.graphics_viewer.setDragMode(QGraphicsView.NoDrag)
    

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W:
            self.graphics_scene.w_key_is_pressed = True
        elif event.key() == Qt.Key_H:
            self.graphics_scene.h_key_is_pressed = True
        elif event.key() == Qt.Key_M:
            self.graphics_scene.m_key_is_pressed = True
        # if event.modifiers() == Qt.ALT:
        elif event.key() == Qt.Key_Z:
            self.graphics_scene.strg_key_is_pressed = True

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_W:
            self.graphics_scene.w_key_is_pressed = False
        elif event.key() == Qt.Key_H:
            self.graphics_scene.h_key_is_pressed = False
        elif event.key() == Qt.Key_M:
            self.graphics_scene.m_key_is_pressed = False
        # if event.modifiers() == Qt.ALT:
        elif event.key() == Qt.Key_Z:
            self.graphics_scene.strg_key_is_pressed = False

        # print("self.w_key_is_pressed =", self.w_key_is_pressed)
        # print("self.h_key_is_pressed =", self.graphics_scene.h_key_is_pressed)
        # super(GraphicsScene, self).keyReleaseEvent(event)
    """

class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(QGraphicsScene(), parent)
        self.drag_draw_button = None
        self.main_window = None
        self._zoom = 0


        self.m_originalX = 0
        self.m_originalY = 0
        self.m_moving = False

        # self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)


    def mousePressEvent(self, event):
        print(event)
        if event.button() == QtCore.Qt.MidButton:
            self.m_originalX = event.x()
            self.m_originalY = event.y()
            self.m_moving = True
            print("self.m_originalX, self.m_originalX, self.m_moving =", self.scene().m_originalX, self.scene().m_originalY, self.scene().m_moving)
        super(GraphicsView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MidButton:
            self.m_moving = False
        super(GraphicsView, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        # print("mouseMoveEvent [GraphicsView] -> ", event.x(), event.y())
        if self.m_moving:
            print("here------------------")
            print("self.m_originalX, self.m_originalY =", self.m_originalX, self.m_originalY)
            try:
                oldp = self.mapToScene(self.m_originalX, self.m_originalY)
                newp = self.mapToScene(event.x(), event.y())
                translation = (newp - oldp)
                print("translation.x(), translation.y() =", translation.x(), translation.y())
                # print("len(self.views()) =", len(self.views()))
                # print("self.views()[0] =", self.views()[0])
                # temp_view = QGraphicsView(self.views()[0])

                # print(self.views()[0]._zoom)

                self.translate(translation.x(), translation.y())

                # self.views()[0] = temp_view
                print("hallo")

                self.m_originalX = event.x()
                self.m_originalY = event.y()
            except Exception as err:
                print("Error:", err)
        super(GraphicsView, self).mouseMoveEvent(event)
    """

    def mousePressEvent(self, event):
        if event.button() == Qt.MidButton:
            self.viewport().setCursor(Qt.ClosedHandCursor)
            self.original_event = event
            handmade_event = QMouseEvent(QEvent.MouseButtonPress, QPointF(event.pos()), Qt.LeftButton, event.buttons(), Qt.KeyboardModifiers())
            self.mousePressEvent(handmade_event)
        super(GraphicsView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MidButton:
            # for changing back to Qt.OpenHandCursor
            self.viewport().setCursor(Qt.OpenHandCursor)
            handmade_event = QMouseEvent(QEvent.MouseButtonRelease, QPointF(event.pos()), Qt.LeftButton, event.buttons(), Qt.KeyboardModifiers())
            self.mouseReleaseEvent(handmade_event)
        super(GraphicsView, self).mouseReleaseEvent(event)

    def hoverMoveEvent(self, event):
        point = event.pos().toPoint()
        print(point)
        QGraphicsItem.hoverMoveEvent(self, event)
        super(GraphicsView, self).hoverMoveEvent(event)
    """


    def fitInView(self, scale=True):
        unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
        self.scale(1.0 / unity.width(), 1.0 / unity.height())
        viewrect = self.viewport().rect()
        scenerect = self.transform().mapRect(self.rect())
        factor = min(viewrect.width() / scenerect.width(),
                     viewrect.height() / scenerect.height())
        self.scale(factor, factor)

    def wheelEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        # point = QPoint(int(event.x()), int(event.y()))
        scene_coords = self.mapToScene(event.x(), event.y())
        x_scene, y_scene = scene_coords.x(), scene_coords.y()
        print("x, y =", x,y)
        print("scene_coords =", scene_coords)
        print("x_scene, y_scene =", x_scene, y_scene)
        # self.scene().pointPressed = (x,y)
        # self.scene().scene_width
        # print("a,b =", a, b)
        # b = max(min(y, self.scene().scene_width), 0)


        # print("self.scene().self.pointPressed =", self.pos().pointPressed)
        if self.scene().w_key_is_pressed or self.scene().h_key_is_pressed:
            self.scene().pointPressed = (x_scene, y_scene)
            increasing = event.angleDelta().y() > 0
            self.scene().change_width_or_height(increasing)
        else:
            # self.graphics_viewer.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            # self.graphics_viewer.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
            print("event.angleDelta().y() =", event.angleDelta().y())
            if event.angleDelta().y() > 0:
                factor = 1.1
                self._zoom += 0.1
            else:
                factor = 0.9
                self._zoom -= 0.1
            self._zoom = round(self._zoom, 1)
            print("self._zoom =", self._zoom)
            if self._zoom > 0:
                print("self._zoom > 0 -> True")
                print("factor =", factor)
                self.scale(factor, factor)
                # self.graphics_viewer.translate(0, -event.angleDelta().y() / 120.0)
                # self.viewport()
            else:
                self.fitInView()
                self._zoom = 0.0
                self.scene().repaint()

            if self.drag_draw_button.isChecked():
                self.setDragMode(QGraphicsView.ScrollHandDrag)
            else:
                self.setDragMode(QGraphicsView.NoDrag)


class Secondary_Window(QMainWindow):

    def __init__(self, main_labeling_tool, mode):
        super(Secondary_Window, self).__init__()
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.labeling_tool = main_labeling_tool
        self.mode = mode

        print("main_labeling_tool =", main_labeling_tool.scale_factor)
        width = 500
        height = 250
        self.setGeometry(200, 200, width, height)
        self.setFixedWidth(width)
        self.setFixedHeight(height)

        self.setWindowTitle("Change Window Size")
        # self.show()
        print("0")
        self.centralwidget = QWidget(self)

        # self.horizontal_Layout = QHBoxLayout()
        # self.vertical_Layout = QVBoxLayout()


        font1 = QFont()
        font1.setPointSize(10)

        self.label_titel = QLabel(self.centralwidget)
        self.label_titel.setFont(font1)
        if mode == "change_window_size":
            self.label_titel.setText("Set a new scaling value: \nRecommanded values are between 0.4 and 1.")


        self.label_titel.setAlignment(QtCore.Qt.AlignCenter)
        label_width = int(width / 3)
        label_height = 50
        # self.label_titel.setFixedWidth(width)
        # self.label_titel.setFixedHeight(label_height)
        # self.label_titel.setGeometry(width/2, height/3, label_width, label_height)

        self.text_line = QLineEdit(self.centralwidget)

        text_line_width = int(width/3)
        text_line_height = 40
        self.text_line.setFont(font1)
        self.text_line.setFixedWidth(text_line_width)
        self.text_line.setFixedHeight(text_line_height)
        # self.text_line.setGeometry(int(width/2 - text_line_width/2), int(height/2 - text_line_height/2), text_line_width, text_line_height)
        if mode == "change_window_size":
            self.text_line.setText(str(self.labeling_tool.scale_factor))
        self.text_line.returnPressed.connect(self.text_line_enter)

        button_width = int(width/4)
        button_height = 40
        self.ok_button = QPushButton("Ok", self.centralwidget)
        self.ok_button.setFixedWidth(button_width)
        self.ok_button.setFixedHeight(button_height)
        # self.ok_button.setGeometry(width-button_width, height-button_height, button_width, button_height)
        self.ok_button.clicked.connect(self.ok_clicked)

        self.setCentralWidget(self.centralwidget)
        self.grid_layout = QGridLayout()


        for col in range(3):
            for row in range(3):
                self.grid_layout.setColumnMinimumWidth(col, int(width/3.0))
                self.grid_layout.setRowMinimumHeight(row, int(height/3.0))
                self.grid_layout.setColumnStretch(col, 10)
                self.grid_layout.setRowStretch(row, 10)



        # self.grid_layout.setColumnMinimumWidth(1, int(width / 3.0))
        # self.grid_layout.setRowMinimumHeight(2, int(height / 3.0))

        self.grid_layout.addWidget(self.label_titel, 0, 0, 1, 3, QtCore.Qt.AlignCenter)
        self.grid_layout.addWidget(self.text_line, 1, 1, alignment=QtCore.Qt.AlignCenter)
        self.grid_layout.addWidget(self.ok_button, 2, 2, alignment=QtCore.Qt.AlignRight)
        self.centralwidget.setLayout(self.grid_layout)

        self.label_titel.move(0, -150)
        # self.label_titel.setGeometry(int(self.label_titel.width()/2), int(height/3), label_width, label_height)
        # self.vertical_Layout.addStretch(1)
        # self.vertical_Layout.addWidget(self.ok_button)



        # self.horizontal_Layout.addStretch(1)
        # self.horizontal_Layout.addWidget(self.text_line)
        # self.horizontal_Layout.addLayout(self.vertical_Layout)

        # self.centralwidget.setLayout(self.horizontal_Layout)


        self.show()


    def check_new_size(self):
        global my_gui
        try:
            temp_factor = float(self.text_line.text())
            self.labeling_tool.close()
            my_gui = Labeling_Tool_GUI(temp_factor)
            self.close()
        except:
            self.text_line.setText(str(self.labeling_tool.scale_factor))

    def change_margin(self):
        global my_gui
        try:
            marging_factor = float(self.text_line.text())
            self.labeling_tool.close()
            scale_factor = self.labeling_tool.scale_factor
            my_gui = Labeling_Tool_GUI(scale_factor, marging_factor)
            self.close()
        except:
            self.text_line.setText(str(self.labeling_tool.scale_factor))

    def add_tag(self):
        global my_gui
        new_tag = self.text_line.text()
        print("new_tag =", new_tag)
        config_file = open(self.labeling_tool.config_path, 'r', encoding="utf-8")
        print("111")
        old_config_file_content = [row for row in config_file.read().splitlines() if not row.split(',')[0] == "tags"]
        config_file.close()
        print("111")
        print(old_config_file_content)

        print("1")
        os.remove(self.labeling_tool.config_path)
        print("2")
        config_file = open(self.labeling_tool.config_path, 'a+', encoding="utf-8")
        print("3")
        self.labeling_tool.tag_list.append(new_tag)

        new_tag_list = self.labeling_tool.tag_list.copy()


        config_file.write("tags," + ','.join(new_tag_list))
        for row in old_config_file_content:
            config_file.write(row)
        config_file.close()


        self.labeling_tool.close()
        scale_factor = self.labeling_tool.scale_factor
        my_gui = Labeling_Tool_GUI(scale_factor)
        self.close()



    def text_line_enter(self):
        global my_gui
        # search for: self.tag_list = sorted(["blurry","covered","noisy"])
        if self.mode == "change_window_size":
            self.check_new_size()
        elif self.mode == "change_margin_sizem":
            self.change_margin()
        elif self.mode == "add_new_tag":
            # filepath = os.getcwd() + os.sep + os.path.basename(__file__)
            self.add_tag()
            """
            config_file = open(self.config_path, 'r', encoding="utf-8")
            config_file = open(filepath, 'r').read().splitlines()

            current_tags = self.labeling_tool.tag_list
            print("current_tags", current_tags)
            print("type(this_file)", type(this_file))
            print("this_file[:10]", this_file[:10])
            """
    def ok_clicked(self):
        global my_gui
        if self.mode == "change_window_size":
            self.check_new_size()
        elif self.mode == "change_margin_size":
            self.change_margin()
        elif self.mode == "add_new_tag":
            pass



class GraphicsScene(QGraphicsScene):
    def __init__(self, scene_width, scene_height):
        super(GraphicsScene, self).__init__()

        self.main_window = None
        self.scene_width, self.scene_height = scene_width, scene_height
        # self.cur_class_color = (0, 163, 232)
        self.cur_class = 0
        """
        self.class_colors = {(0, 163, 232):0,
                             (0, 200, 0):1,
                             (232, 163, 0):2,
                             (200, 0, 0):3
                             }
        """
        self.class_colors = [(0, 163, 232),
                             (0, 200, 0),
                             (232, 163, 0),
                             (200, 0, 0)
                             ]

        # These values will be pointers to values in the GUI. So, there should be no two different tag-lists
        self.chk_box_list = None
        self.tag_list = None
        self.combo_box = None

        # These values will change during the normal usage, but should be reset if we save all the labels.
        self.is_classifing = True  # True => Classification Mode; False => Tagging Mode
        self.temp_coord = []
        self.coord_list = []
        self.highlighted_person_id_in_coord_list = None
        self.w_key_is_pressed = False
        self.h_key_is_pressed = False
        self.m_key_is_pressed = False
        self.drag_on = False

        self.circle_radius = 4.0
        self.pen_width = 2.0
        self.pen_width_highlighting = 4
        self.pointPressed = QPointF()
        self.pointReleased = QPointF()

        # how much should the width and height increase or decrease if we change the width of height with mouse wheel
        self.step_width = 1.0

        self.m_originalX = 0
        self.m_originalY = 0
        self.m_moving = False

        self.start_x = 0
        self.start_y = 0
        self.end_x = scene_width
        self.end_y = scene_height
        self.setSceneRect(self.start_x, self.start_y, self.end_x, self.end_y)
        """
        pixmap = QPixmap(100, 100)
        pixmap.fill(QtCore.Qt.red)

        self.pixmap_item = self.addPixmap(pixmap)
        
        # random position
        # self.pixmap_item.setPos(*random.sample(range(-100, 100), 2))
        """





    def reset_values(self):
        self.is_classifing = True  # True => Classification Mode; False => Tagging Mode
        self.temp_coord = []
        self.coord_list = []
        self.highlighted_person_id_in_coord_list = None
        self.w_key_is_pressed = False
        self.h_key_is_pressed = False
        self.m_key_is_pressed = False
        self.strg_key_is_pressed = False

        self.pointPressed = QPointF()
        self.pointReleased = QPointF()

        self.setSceneRect(self.start_x, self.start_y, self.end_x, self.end_y)

        self.update()



    def set_checkboxes(self, marked_person_id):
        used_tags = []
        for idx, elem in enumerate(self.coord_list):
            if idx == marked_person_id:
                if len(elem) == 2:
                    # Uncheck all checkboxes
                    for chk_box in self.chk_box_list:
                        chk_box.setChecked(False)
                else:
                    cls, coord, tags = elem
                    for chk_box in self.chk_box_list:
                        if chk_box.text() in tags:
                            chk_box.setChecked(True)
                        else:
                            chk_box.setChecked(False)



        return used_tags

    def combo_box_method(self):
        """
        If we select a bounding box, then this method changes the current chosen person class
        in the combo_box to the class of the selected person. So, with this method the user
        get the class-information of the current selected person.

        :return:
        """
        if not self.highlighted_person_id_in_coord_list is None:
            person_data = self.coord_list[self.highlighted_person_id_in_coord_list]
            print("person_data =", person_data)
            self.combo_box.setCurrentIndex(person_data[0])


    def repaint(self):

        for elem in self.items():
            if isinstance(elem, QGraphicsRectItem) or isinstance(elem, QGraphicsEllipseItem):
                self.removeItem(elem)

        for idx, elem in enumerate(self.coord_list):
            if len(elem) == 2:
                cls, coord = elem
            else:
                cls, coord, tags = elem

            start_x = coord[0][0]
            start_y = coord[0][1]
            width = coord[1][0] - start_x
            height = coord[1][1] - start_y
            self.cur_class = cls
            if idx == self.highlighted_person_id_in_coord_list:
                self.addEllipse(round(start_x - (self.circle_radius / 2.0),0), round(start_y - (self.circle_radius / 2.0),0), self.circle_radius, self.circle_radius, self.get_pen(is_highlighted=True))
                self.addEllipse(round(start_x+width - (self.circle_radius / 2.0),0), round(start_y+height - (self.circle_radius / 2.0),0), self.circle_radius, self.circle_radius, self.get_pen(is_highlighted=True))
                self.addRect(start_x, start_y, width, height, self.get_pen(is_highlighted=True))
            else:
                self.addEllipse(round(start_x - (self.circle_radius / 2.0),0), round(start_y - (self.circle_radius / 2.0),0), self.circle_radius, self.circle_radius, self.get_pen(is_highlighted=False))
                self.addEllipse(round(start_x+width - (self.circle_radius / 2.0),0), round(start_y+height - (self.circle_radius / 2.0),0), self.circle_radius, self.circle_radius, self.get_pen(is_highlighted=False))
                self.addRect(start_x, start_y, width, height, self.get_pen(is_highlighted=False))



    def get_pen(self, is_highlighted=False):
        c1, c2, c3 = self.class_colors[self.cur_class][0], self.class_colors[self.cur_class][1], self.class_colors[self.cur_class][2]

        if is_highlighted:
            return QPen(QColor(c1, c2, c3), self.pen_width_highlighting)
        else:
            return QPen(QColor(c1, c2, c3), self.pen_width)

    def get_nearest_Box(self):
        x_mouse,y_mouse = self.pointPressed

        min_dist = 10 ** 100
        min_coordinates = None
        min_elem_QT_item = None

        for idx, elem in enumerate(self.items()):
            if isinstance(elem, QGraphicsRectItem):
                print("get_nearest_Box -> elem ->", elem)
                """
                x_box is the x-coordinate of the rectangle's left edge
                    -> see: https://doc.qt.io/qt-5/qrectf.html#x
                y_box is the y-coordinate of the rectangle's top edge
                    -> see: https://doc.qt.io/qt-5/qrectf.html#y
                """
                x_box, y_box, w_box, h_box = elem.rect().x(), elem.rect().y(), elem.rect().width(), elem.rect().height()
                print("x_box, y_box, w_box, h_box =", x_box, y_box, w_box, h_box)
                x_center = int(x_box + (w_box / 2.0))
                y_center = int(y_box + (h_box / 2.0))
                cur_dist = np.sqrt((x_center - x_mouse) ** 2 + (y_center - y_mouse) ** 2)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    min_elem_QT_item = elem
                    min_coordinates = (x_box, y_box, x_box+w_box, y_box+h_box)
                    # min_idx = idx


        print("get_nearest_Box -> min_dist ->", min_dist)
        print("get_nearest_Box -> min_coordinates (before) ->", min_coordinates)
        min_coordinates_copy = [round(coord,0) for coord in min_coordinates]
        circle_elements = []
        print("get_nearest_Box -> min_coordinates_copy (before, after) ->", min_coordinates_copy)
        for idx, elem in enumerate(self.items()):
            if isinstance(elem, QGraphicsEllipseItem):
                # print("elem.ellipse() =", elem.rect())
                x_circ, y_circ = elem.rect().x(), elem.rect().y()
                # print("elem.ellipse() -> x_circ, y_circ =", x_circ, y_circ)
                print("elem.ellipse() -> x_circ+self.circle_radius/2.0 =", x_circ+self.circle_radius/2.0)
                print("elem.ellipse() -> y_circ+self.circle_radius/2.0 =", y_circ+self.circle_radius/2.0)
                if round(min_coordinates[0],0) == round(x_circ+(self.circle_radius/2.0),0) and round(min_coordinates[1],0) == round(y_circ+(self.circle_radius/2.0),0):
                    min_coordinates = (-1, -1, min_coordinates[2], min_coordinates[3])
                    circle_elements.append(elem)
                if round(min_coordinates[2],0) == round(x_circ+(self.circle_radius/2.0),0) and round(min_coordinates[3],0) == round(y_circ+(self.circle_radius/2.0),0):
                    min_coordinates = (min_coordinates[0], min_coordinates[1], -1, -1)
                    circle_elements.append(elem)
        print("get_nearest_Box -> min_coordinates (after) ->", min_coordinates)

        idx_min_elem_coord_list = None
        min_coordinates = min_coordinates_copy
        if not min_elem_QT_item is None:
            for idx, elem in enumerate(self.coord_list):
                print("elem[1] =", elem[1])
                elem = [(round(coord[0],0), round(coord[1],0)) for coord in elem[1]]
                print("get_nearest_Box -> elem =?= min_coordinates ->", elem, min_coordinates)
                if int(elem[0][0]) == int(min_coordinates[0]) and int(elem[0][1]) == int(min_coordinates[1]) and int(elem[1][0]) == int(min_coordinates[2]) and int(elem[1][1]) == int(min_coordinates[3]):
                    idx_min_elem_coord_list = idx
                    break

        return min_elem_QT_item, idx_min_elem_coord_list, circle_elements

    def delete_nearest_Box(self):
        print("self.coord_list [before]=", self.coord_list)
        print("self.pointPressed =", self.pointPressed)
        min_elem_QT_item, idx_min_elem_coord_list, circle_elements = self.get_nearest_Box()
        print("self.coord_list =", self.coord_list)
        print("min_elem_QT_item, idx_min_elem_coord_list, circle_elements =", min_elem_QT_item, idx_min_elem_coord_list, circle_elements)
        print("circle_elements =", circle_elements)
        if not min_elem_QT_item is None:
            self.removeItem(min_elem_QT_item)
            if circle_elements[0] is None:
                print("circle_elements[0] is None:")
            else:
                print("len(self.items()) [before]", len(self.items()))
                self.removeItem(circle_elements[0])
                print("len(self.items()) [after]", len(self.items()))
            if circle_elements[1] is None:
                print("circle_elements[0] is None:")
            else:
                print("len(self.items()) [before]", len(self.items()))
                self.removeItem(circle_elements[1])
                print("len(self.items()) [after]", len(self.items()))
            del self.coord_list[idx_min_elem_coord_list]

        print("self.coord_list [after]=", self.coord_list)




    def reorder_circles(self):
        """
        The idea of this function is, that the user has four possibilities to draw a box
            * top-left - bottom-right
            * bottom-right - top-left
            * top-right - bottom-left
            * bottom-left - top-right
        So, self.temp_coord would not be consistent. To be consistent we reorder the self.temp_coord
        as top-left - bottom-right. We also delete the circles and draw them in the new order.
        :return:
        """
        old_temp_coord = self.temp_coord.copy()
        print("old_temp_coord =", old_temp_coord)
        # Make sure, that the coordinates in self.temp_coord are ordered top-left and bottom-right
        x_left = round(min(self.temp_coord[0][0], self.temp_coord[1][0]),0)
        y_top = round(min(self.temp_coord[0][1], self.temp_coord[1][1]),0)
        x_right = round(max(self.temp_coord[0][0], self.temp_coord[1][0]),0)
        y_bottom = round(max(self.temp_coord[0][1], self.temp_coord[1][1]),0)
        self.temp_coord = [(x_left, y_top),(x_right,y_bottom)]

        # Delete the circles
        for elem in self.items():
            if isinstance(elem, QGraphicsEllipseItem):
                x_circ, y_circ = elem.rect().x(), elem.rect().y()
                if old_temp_coord[0][0] == round(x_circ + self.circle_radius / 2.0, 0) and old_temp_coord[0][1] == round(y_circ + self.circle_radius / 2.0, 0):
                    old_temp_coord = [(-1, -1), (old_temp_coord[1][0], old_temp_coord[1][1])]
                    self.removeItem(elem)
                if old_temp_coord[1][0] == round(x_circ + self.circle_radius / 2.0, 0) and old_temp_coord[1][1] == round(y_circ + self.circle_radius / 2.0, 0):
                    old_temp_coord = [(old_temp_coord[0][0], old_temp_coord[0][1]), (-1, -1)]
                    self.removeItem(elem)
        print("self.temp_coord =", self.temp_coord)
        # Draw the Circles in the new order
        pen = self.get_pen()
        self.addEllipse(round(x_left - self.circle_radius / 2.0, 0), round(y_top - self.circle_radius / 2.0, 0), self.circle_radius, self.circle_radius, pen)
        self.addEllipse(round(x_right - self.circle_radius / 2.0, 0), round(y_bottom - self.circle_radius / 2.0, 0), self.circle_radius, self.circle_radius, pen)

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton and self.is_classifing and not self.drag_on:  # left button pressed
            print("type(event) =", type(event))
            # self.pointPressed = (event.scenePos().x(), event.scenePos().y())
            self.pointPressed = (max(min(event.scenePos().x(), self.scene_width),0), max(min(event.scenePos().y(), self.scene_height),0))
            self.temp_coord.append(self.pointPressed)
            pen = self.get_pen()
            # brush = QBrush(QColor(self.class_colors_GS[self.cur_class_GS]))


            # TODO: Draw a circle for the current point
            # ellipse = self.addellipse(10, 10, 200, 200)
            x = float(self.pointPressed[0])
            y = float(self.pointPressed[1])

            self.addEllipse(round(x-(self.circle_radius/2.0),0), round(y-(self.circle_radius/2.0),0), self.circle_radius, self.circle_radius, pen)

            print("self.temp_coord =", self.temp_coord)
            print("self.coord_list =", self.coord_list)
            if len(self.temp_coord) == 2:
                # First, delete the circles which correspond to this box and draw two new circles top-left and bottom-right
                self.reorder_circles()

                # TODO: Draw Bounding Box
                start_x = round(float(min(self.temp_coord[0][0], self.temp_coord[1][0])),0)
                start_y = round(float(min(self.temp_coord[0][1], self.temp_coord[1][1])),0)
                width = round(float(max(self.temp_coord[0][0], self.temp_coord[1][0]) - min(self.temp_coord[0][0], self.temp_coord[1][0])),0)
                height = round(float(max(self.temp_coord[0][1], self.temp_coord[1][1]) - min(self.temp_coord[0][1], self.temp_coord[1][1])),0)
                print("start_x, start_y, width, height =", start_x, start_y, width, height)
                # self.addRect(self.temp_coord[0][0], self.temp_coord[0][1], width, height, pen, brush)
                # path = QPainterPath()
                # path.addRect(start_x, start_y, width, height)
                # self.addPath(path, pen)
                self.addRect(start_x, start_y, width, height, pen)

                # self.addRect(self.temp_coord[0][0], self.temp_coord[0][1], width, height, pen, brush)
                print("self.temp_coord (before) =", self.temp_coord)
                self.temp_coord = [(round(coord[0],0), round(coord[1],0)) for coord in self.temp_coord]
                print("self.temp_coord (after) =", self.temp_coord)
                self.coord_list.append([self.cur_class, self.temp_coord])
                self.temp_coord = []
        elif event.buttons () == QtCore.Qt.RightButton and self.is_classifing: # right click
            print("Right click")
            self.pointPressed = (event.scenePos().x(), event.scenePos().y())
            self.delete_nearest_Box()
            self.repaint()
        elif event.buttons() == QtCore.Qt.LeftButton and not self.is_classifing:  # left button pressed
            # ToDo: Find the nearest Person and highlight this person by increasing the box-thickness
            self.pointPressed = (event.scenePos().x(), event.scenePos().y())
            print("len(self.items()) =", len(self.items()))
            min_elem_QT_item, idx_min_elem_coord_list, circle_elements = self.get_nearest_Box()
            self.highlighted_person_id_in_coord_list = idx_min_elem_coord_list
            print("len(self.items()) =", len(self.items()))

            self.set_checkboxes(idx_min_elem_coord_list)
            self.combo_box_method()

            # cls, coord = self.coord_list[idx_min_elem_coord_list]
            # self.cur_class = cls

            self.repaint()



            """
            for elem in self.items():
                if isinstance(elem, QRect):
                    elem.setPen(self.get_pen(is_highlighted=False))
            """

            """
            if self.last_marked_elem_QT is None:
                min_elem_QT_item.setPen(self.get_pen(is_highlighted=True))
                # self.removeItem(min_elem_QT_item)
                # print("min_elem_QT_item =", type(min_elem_QT_item))
                # print("min_elem_QT_item =", min_elem_QT_item.rect())
                print("min_elem_QT_item =", type(self))
            else:
                self.cur_class = self.last_marked_elem_cls
                
            """

        """
        elif event.buttons() == QtCore.Qt.MidButton:
            self.m_originalX = event.x()
            self.m_originalY = event.y()
            self.m_moving = True
            print("self.m_originalX, self.m_originalX, self.m_moving =", self.m_originalX, self.m_originalY, self.m_moving)
        """











        # print('pointPressed: {}'.format(self.pointPressed))

        """
        items = self.items(event.scenePos())
        for item in items:
            print("item =", item)
            if item is self.pixmap_item:
                print(item.mapFromScene(event.scenePos()))
        """

        # super(GraphicsScene, self).mousePressEvent(event)
    """
    def mouseReleaseEvent(self, event):
        if event.buttons() == QtCore.Qt.MidButton:
            self.m_moving = False
    """

    # This Method prints the mouse position without click (just hovering)
    def mouseMoveEvent(self, event):
        # print("Mouse Position ", event.scenePos())
        # print("self.hasFocus() =", self.hasFocus())
        # print("self.stickyFocus() =", self.stickyFocus())
        # print("event.buttonDownScenePos(QtCore.Qt.MidButton) ", event.buttonDownScenePos(QtCore.Qt.MidButton))
        # print("self.m_key_is_pressed =", self.m_key_is_pressed)
        # print("self.main_window.text_edit.hasFocus() =", self.main_window.text_edit.hasFocus())
        self.main_window.setFocus()
        if event.scenePos().x() < 0 or event.scenePos().y() < 0 or self.scene_width < event.scenePos().x() or self.scene_height < event.scenePos().y():
            try:
                # print("mouseMoveEvent [GraphicsScene] -> ", event.scenePos().x(), event.scenePos().y())
                self.views()[0].fitInView()
            except Exception as err:
                print("err =", err)


        if self.m_key_is_pressed:
        # if True:
            # if self.mouse_wheel_pressed:
            self.pointPressed = (event.scenePos().x(), event.scenePos().y())
            min_elem_QT_item, idx_min_elem_coord_list, circle_elements = self.get_nearest_Box()
            print("min_elem_QT_item, idx_min_elem_coord_list, circle_elements =", min_elem_QT_item, idx_min_elem_coord_list, circle_elements)
            print("self.pointPressed =", self.pointPressed)
            coord = self.coord_list[idx_min_elem_coord_list][1]

            width = coord[1][0] - coord[0][0]
            height = coord[1][1] - coord[0][1]
            # x_center = coord[0][0] + width / 2.0
            # y_center = coord[0][1] + height / 2.0

            x_center_new = event.scenePos().x()
            y_center_new = event.scenePos().y()

            x_left = x_center_new - width / 2.0
            x_right = x_center_new + width / 2.0
            y_top = y_center_new - height / 2.0
            y_bottom = y_center_new + height / 2.0

            too_left = x_left < 0
            too_right = x_right > self.width()
            too_above = y_top < 0
            too_below = y_bottom > self.height()

            if not (too_left or too_right or too_above or too_below):
                self.coord_list[idx_min_elem_coord_list][1] = [(x_left, y_top), (x_right, y_bottom)]
            self.repaint()

            # self.repaint()
        if self.m_moving:
            """
            try:
                print("here------------------")
                print("self.m_originalX, self.m_originalY =", self.m_originalX, self.m_originalY)
                oldp = self.views()[0].mapToScene(self.m_originalX, self.m_originalY)
                print("len(self.views()) =", len( self.views()))
                newp = self.views()[0].mapToScene(event.scenePos().x(), event.scenePos().y())
                translation = 0.01 * (newp - oldp)
                print("translation.x(), translation.y() =", translation.x(), translation.y())
                # print("len(self.views()) =", len(self.views()))
                # print("self.views()[0] =", self.views()[0])
                # temp_view = QGraphicsView(self.views()[0])

                # print(self.views()[0]._zoom)


                self.views()[0].translate(translation.x(), translation.y())



                # self.views()[0] = temp_view
                print("hallo")

                self.m_originalX = event.scenePos().x()
                self.m_originalY = event.scenePos().y()
            except Exception as err:
                # logf.write(str(err))
                print("Error:", err)
            # self.update()
            """


    """
    # This works
    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == Qt.Key_W:
                self.w_key_is_pressed = True
            elif event.key() == Qt.Key_H:
                self.h_key_is_pressed = True
            elif event.key() == Qt.Key_Z:
                self.strg_key_is_pressed = True
            elif event.key() == Qt.Key_M:
                self.m_key_is_pressed = True



        # print("self.m_key_is_pressed =", self.m_key_is_pressed)
        # print("self.w_key_is_pressed =", self.w_key_is_pressed)
        # print("self.h_key_is_pressed =", self.h_key_is_pressed)
        super(GraphicsScene, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == Qt.Key_W:
                self.w_key_is_pressed = False
            elif event.key() == Qt.Key_H:
                self.h_key_is_pressed = False
            elif event.key() == Qt.Key_M:
                self.m_key_is_pressed = False
            # print("self.m_key_is_pressed =", self.m_key_is_pressed)


        # print("self.w_key_is_pressed =", self.w_key_is_pressed)
        # print("self.h_key_is_pressed =", self.h_key_is_pressed)
        super(GraphicsScene, self).keyReleaseEvent(event)
    """


    def change_width_or_height(self, increasing):
        min_elem_QT_item, idx_min_elem_coord_list, circle_elements = self.get_nearest_Box()
        coord = self.coord_list[idx_min_elem_coord_list][1]

        too_wide = (coord[0][0] - self.step_width) < 0 or self.width() < (coord[1][0] + self.step_width)
        too_small_width = (coord[1][0] - self.step_width) - (coord[0][0] + self.step_width) <= 0
        too_high = (coord[0][1] - self.step_width) < 0 or self.height() < (coord[1][1] + self.step_width)
        too_small_height = (coord[1][1] - self.step_width) - (coord[0][1] + self.step_width) <= 0
        print("too_wide =", too_wide)
        print("too_small_width =", too_small_width)
        print("too_high =", too_high)
        print("too_small_height =", too_small_height)
        if self.w_key_is_pressed and increasing and not too_wide:
            self.coord_list[idx_min_elem_coord_list][1] = [(self.coord_list[idx_min_elem_coord_list][1][0][0] - self.step_width, self.coord_list[idx_min_elem_coord_list][1][0][1]),
                                                           (self.coord_list[idx_min_elem_coord_list][1][1][0] + self.step_width, self.coord_list[idx_min_elem_coord_list][1][1][1])]
        elif self.w_key_is_pressed and not increasing and not too_small_width:
            self.coord_list[idx_min_elem_coord_list][1] = [(self.coord_list[idx_min_elem_coord_list][1][0][0] + self.step_width, self.coord_list[idx_min_elem_coord_list][1][0][1]),
                                                           (self.coord_list[idx_min_elem_coord_list][1][1][0] - self.step_width, self.coord_list[idx_min_elem_coord_list][1][1][1])]
        if self.h_key_is_pressed and increasing and not too_high:
            self.coord_list[idx_min_elem_coord_list][1] = [(self.coord_list[idx_min_elem_coord_list][1][0][0], self.coord_list[idx_min_elem_coord_list][1][0][1] - self.step_width),
                                                           (self.coord_list[idx_min_elem_coord_list][1][1][0], self.coord_list[idx_min_elem_coord_list][1][1][1] + self.step_width)]
        elif self.h_key_is_pressed and not increasing and not too_small_height:
            self.coord_list[idx_min_elem_coord_list][1] = [(self.coord_list[idx_min_elem_coord_list][1][0][0], self.coord_list[idx_min_elem_coord_list][1][0][1] + self.step_width),
                                                           (self.coord_list[idx_min_elem_coord_list][1][1][0], self.coord_list[idx_min_elem_coord_list][1][1][1] - self.step_width)]
        self.repaint()

    def wheelEvent(self, event):
        print("event =", event.delta()/120)
        self.pointPressed = (event.scenePos().x(), event.scenePos().y())
        print("self.pointPressed =", self.pointPressed)
        print("self.self.w_key_is_pressed or self.h_key_is_pressed =", self.w_key_is_pressed or self.h_key_is_pressed)
        increasing = event.delta() > 0
        if self.w_key_is_pressed or self.h_key_is_pressed:
            self.change_width_or_height(increasing)

    """
        def mousePressEvent(self, event):
        print(event)
        if event.button() == QtCore.Qt.MidButton:
            self.m_originalX = event.x()
            self.m_originalY = event.y()
            self.m_moving = True
            print("self.m_originalX, self.m_originalX, self.m_moving =", self.m_originalX, self.m_originalY, self.m_moving)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MidButton:
            self.m_moving = False


    def mouseMoveEvent(self, event: QMouseEvent):
        # print("type(event) =", type(event))
        if self.m_moving:
            print("here------------------")
            oldp = self.mapToScene(self.m_originalX, self.m_originalY)
            newp = self.mapToScene(event.pos())
            translation = newp - oldp
            print("translation.x(), translation.y() =", translation.x(), translation.y())
            self.translate(translation.x(), translation.y())

            self.m_originalX = event.x()
            self.m_originalY = event.y()

            self.update()
    """





app = QApplication(sys.argv)
my_gui = Labeling_Tool_GUI(scale_factor=0.9)
sys.exit(app.exec_())