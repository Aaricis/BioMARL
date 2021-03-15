# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\dell\Desktop\TD-FALCON\DQN_version\TD_Falcon.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt
import os
import sys
from math import sin,cos,pi


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1000, 700)
        Form.setMinimumSize(QtCore.QSize(1000, 700))
        Form.setMaximumSize(QtCore.QSize(1000, 700))
        self.MineField = QtWidgets.QGroupBox(Form)
        self.MineField.setGeometry(QtCore.QRect(0, 0, 500, 475))
        self.MineField.setStyleSheet("")
        self.MineField.setObjectName("MineField")
        self.Result = QtWidgets.QTextBrowser(self.MineField)
        self.Result.setGeometry(QtCore.QRect(0, 450, 500, 25))
        self.Result.setObjectName("Result")
        self.Automatic_Control = QtWidgets.QGroupBox(Form)
        self.Automatic_Control.setGeometry(QtCore.QRect(0, 475, 500, 225))
        self.Automatic_Control.setObjectName("Automatic_Control")
        self.Button_Reset = QtWidgets.QPushButton(self.Automatic_Control)
        self.Button_Reset.setGeometry(QtCore.QRect(60, 20, 75, 23))
        self.Button_Reset.setIconSize(QtCore.QSize(16, 16))
        self.Button_Reset.setObjectName("Button_Reset")
        self.Button_Step = QtWidgets.QPushButton(self.Automatic_Control)
        self.Button_Step.setGeometry(QtCore.QRect(160, 20, 75, 23))
        self.Button_Step.setObjectName("Button_Step")
        self.Button_Auto = QtWidgets.QPushButton(self.Automatic_Control)
        self.Button_Auto.setGeometry(QtCore.QRect(260, 20, 75, 23))
        self.Button_Auto.setObjectName("Button_Auto")
        self.Experiment_Settings = QtWidgets.QGroupBox(self.Automatic_Control)
        self.Experiment_Settings.setGeometry(QtCore.QRect(10, 40, 241, 101))
        self.Experiment_Settings.setObjectName("Experiment_Settings")
        self.ExperimentSetting = QtWidgets.QGridLayout(self.Experiment_Settings)
        self.ExperimentSetting.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.ExperimentSetting.setObjectName("ExperimentSetting")
        self.E4 = QtWidgets.QLabel(self.Experiment_Settings)
        self.E4.setObjectName("E4")
        self.ExperimentSetting.addWidget(self.E4, 1, 2, 1, 1)
        self.E1 = QtWidgets.QLabel(self.Experiment_Settings)
        self.E1.setObjectName("E1")
        self.ExperimentSetting.addWidget(self.E1, 0, 0, 1, 1)
        self.E3 = QtWidgets.QLabel(self.Experiment_Settings)
        self.E3.setObjectName("E3")
        self.ExperimentSetting.addWidget(self.E3, 0, 2, 1, 1)
        self.Target_Moving = QtWidgets.QCheckBox(self.Experiment_Settings)
        self.Target_Moving.setText("")
        self.Target_Moving.setObjectName("Target_Moving")
        self.ExperimentSetting.addWidget(self.Target_Moving, 1, 3, 1, 1, QtCore.Qt.AlignHCenter)
        self.Agent_num = QtWidgets.QLineEdit(self.Experiment_Settings)
        self.Agent_num.setObjectName("Agent_num")
        self.ExperimentSetting.addWidget(self.Agent_num, 0, 1, 1, 1)
        self.E2 = QtWidgets.QLabel(self.Experiment_Settings)
        self.E2.setObjectName("E2")
        self.ExperimentSetting.addWidget(self.E2, 1, 0, 1, 1)
        self.Steps = QtWidgets.QLineEdit(self.Experiment_Settings)
        self.Steps.setObjectName("Steps")
        self.ExperimentSetting.addWidget(self.Steps, 0, 3, 1, 1)
        self.Trials = QtWidgets.QLineEdit(self.Experiment_Settings)
        self.Trials.setObjectName("Trials")
        self.ExperimentSetting.addWidget(self.Trials, 1, 1, 1, 1)
        self.View = QtWidgets.QGroupBox(self.Automatic_Control)
        self.View.setGeometry(QtCore.QRect(10, 140, 241, 52))
        self.View.setObjectName("View")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.View)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.TimeDelay = QtWidgets.QLabel(self.View)
        self.TimeDelay.setObjectName("TimeDelay")
        self.horizontalLayout.addWidget(self.TimeDelay)
        self.V1 = QtWidgets.QLineEdit(self.View)
        self.V1.setObjectName("V1")
        self.horizontalLayout.addWidget(self.V1)
        self.V2 = QtWidgets.QLabel(self.View)
        self.V2.setObjectName("V2")
        self.horizontalLayout.addWidget(self.V2)
        self.Traced = QtWidgets.QCheckBox(self.View)
        self.Traced.setText("")
        self.Traced.setObjectName("Traced")
        self.horizontalLayout.addWidget(self.Traced)
        self.Learnin_Parameters = QtWidgets.QGroupBox(self.Automatic_Control)
        self.Learnin_Parameters.setGeometry(QtCore.QRect(260, 50, 211, 141))
        self.Learnin_Parameters.setObjectName("Learnin_Parameters")
        self.gridLayout = QtWidgets.QGridLayout(self.Learnin_Parameters)
        self.gridLayout.setObjectName("gridLayout")
        self.L2 = QtWidgets.QLabel(self.Learnin_Parameters)
        self.L2.setObjectName("L2")
        self.gridLayout.addWidget(self.L2, 1, 0, 1, 1)
        self.L1 = QtWidgets.QLabel(self.Learnin_Parameters)
        self.L1.setObjectName("L1")
        self.gridLayout.addWidget(self.L1, 0, 0, 1, 1)
        self.Immediate_Reward = QtWidgets.QCheckBox(self.Learnin_Parameters)
        self.Immediate_Reward.setText("")
        self.Immediate_Reward.setObjectName("Immediate_Reward")
        self.gridLayout.addWidget(self.Immediate_Reward, 0, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.Bounde_TD_Rule = QtWidgets.QCheckBox(self.Learnin_Parameters)
        self.Bounde_TD_Rule.setText("")
        self.Bounde_TD_Rule.setObjectName("Bounde_TD_Rule")
        self.gridLayout.addWidget(self.Bounde_TD_Rule, 1, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.Button_Patrol = QtWidgets.QPushButton(self.Automatic_Control)
        self.Button_Patrol.setGeometry(QtCore.QRect(360, 20, 75, 23))
        self.Button_Patrol.setObjectName("Button_Patrol")
        self.Sonar_Signal_Input = QtWidgets.QGroupBox(Form)
        self.Sonar_Signal_Input.setGeometry(QtCore.QRect(500, 0, 500, 151))
        self.Sonar_Signal_Input.setObjectName("Sonar_Signal_Input")
        self.Sonar_Input_5 = QtWidgets.QFrame(self.Sonar_Signal_Input)
        self.Sonar_Input_5.setGeometry(QtCore.QRect(400, 50, 100, 100))
        self.Sonar_Input_5.setFrameShape(QtWidgets.QFrame.Box)
        self.Sonar_Input_5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Sonar_Input_5.setObjectName("Sonar_Input_5")
        self.Sonar_Input_3 = QtWidgets.QFrame(self.Sonar_Signal_Input)
        self.Sonar_Input_3.setGeometry(QtCore.QRect(200, 50, 100, 100))
        self.Sonar_Input_3.setFrameShape(QtWidgets.QFrame.Box)
        self.Sonar_Input_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Sonar_Input_3.setObjectName("Sonar_Input_3")
        self.Sonar_Input_4 = QtWidgets.QFrame(self.Sonar_Signal_Input)
        self.Sonar_Input_4.setGeometry(QtCore.QRect(300, 50, 100, 100))
        self.Sonar_Input_4.setFrameShape(QtWidgets.QFrame.Box)
        self.Sonar_Input_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Sonar_Input_4.setObjectName("Sonar_Input_4")
        self.Sonar_Input_2 = QtWidgets.QFrame(self.Sonar_Signal_Input)
        self.Sonar_Input_2.setGeometry(QtCore.QRect(100, 50, 100, 100))
        self.Sonar_Input_2.setFrameShape(QtWidgets.QFrame.Box)
        self.Sonar_Input_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Sonar_Input_2.setObjectName("Sonar_Input_2")
        self.Sonar_Input_1 = QtWidgets.QFrame(self.Sonar_Signal_Input)
        self.Sonar_Input_1.setGeometry(QtCore.QRect(0, 50, 100, 100))
        self.Sonar_Input_1.setFrameShape(QtWidgets.QFrame.Box)
        self.Sonar_Input_1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Sonar_Input_1.setObjectName("Sonar_Input_1")
        self.AV_Sonar_Signal = QtWidgets.QGroupBox(Form)
        self.AV_Sonar_Signal.setGeometry(QtCore.QRect(500, 200, 500, 151))
        self.AV_Sonar_Signal.setObjectName("AV_Sonar_Signal")
        self.AV_Sonar_5 = QtWidgets.QFrame(self.AV_Sonar_Signal)
        self.AV_Sonar_5.setGeometry(QtCore.QRect(400, 50, 100, 100))
        self.AV_Sonar_5.setFrameShape(QtWidgets.QFrame.Box)
        self.AV_Sonar_5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.AV_Sonar_5.setObjectName("AV_Sonar_5")
        self.AV_Sonar_3 = QtWidgets.QFrame(self.AV_Sonar_Signal)
        self.AV_Sonar_3.setGeometry(QtCore.QRect(200, 50, 100, 100))
        self.AV_Sonar_3.setFrameShape(QtWidgets.QFrame.Box)
        self.AV_Sonar_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.AV_Sonar_3.setObjectName("AV_Sonar_3")
        self.AV_Sonar_4 = QtWidgets.QFrame(self.AV_Sonar_Signal)
        self.AV_Sonar_4.setGeometry(QtCore.QRect(300, 50, 100, 100))
        self.AV_Sonar_4.setFrameShape(QtWidgets.QFrame.Box)
        self.AV_Sonar_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.AV_Sonar_4.setObjectName("AV_Sonar_4")
        self.AV_Sonar_2 = QtWidgets.QFrame(self.AV_Sonar_Signal)
        self.AV_Sonar_2.setGeometry(QtCore.QRect(100, 50, 100, 100))
        self.AV_Sonar_2.setFrameShape(QtWidgets.QFrame.Box)
        self.AV_Sonar_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.AV_Sonar_2.setObjectName("AV_Sonar_2")
        self.AV_Sonar_1 = QtWidgets.QFrame(self.AV_Sonar_Signal)
        self.AV_Sonar_1.setGeometry(QtCore.QRect(0, 50, 100, 100))
        self.AV_Sonar_1.setFrameShape(QtWidgets.QFrame.Box)
        self.AV_Sonar_1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.AV_Sonar_1.setObjectName("AV_Sonar_1")
        self.Current_and_TargetBearing = QtWidgets.QGroupBox(Form)
        self.Current_and_TargetBearing.setGeometry(QtCore.QRect(500, 400, 500, 200))
        self.Current_and_TargetBearing.setObjectName("Current_and_TargetBearing")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.Current_and_TargetBearing)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.T1 = QtWidgets.QLabel(self.Current_and_TargetBearing)
        self.T1.setMaximumSize(QtCore.QSize(16777215, 26))
        self.T1.setAlignment(QtCore.Qt.AlignCenter)
        self.T1.setObjectName("T1")
        self.gridLayout_2.addWidget(self.T1, 0, 0, 1, 1)
        self.T2 = QtWidgets.QLabel(self.Current_and_TargetBearing)
        self.T2.setMaximumSize(QtCore.QSize(16777215, 26))
        self.T2.setAlignment(QtCore.Qt.AlignCenter)
        self.T2.setObjectName("T2")
        self.gridLayout_2.addWidget(self.T2, 0, 1, 1, 1)
        self.Current_Bearing = QtWidgets.QFrame(self.Current_and_TargetBearing)
        self.Current_Bearing.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Current_Bearing.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Current_Bearing.setObjectName("Current_Bearing")
        self.gridLayout_2.addWidget(self.Current_Bearing, 1, 0, 1, 1)
        self.Target_Bearing = QtWidgets.QFrame(self.Current_and_TargetBearing)
        self.Target_Bearing.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Target_Bearing.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Target_Bearing.setObjectName("Target_Bearing")
        self.gridLayout_2.addWidget(self.Target_Bearing, 1, 1, 1, 1)
        self.Mannual_Control = QtWidgets.QGroupBox(Form)
        self.Mannual_Control.setGeometry(QtCore.QRect(500, 610, 500, 80))
        self.Mannual_Control.setObjectName("Mannual_Control")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.Mannual_Control)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.Button_LeftArrow = QtWidgets.QToolButton(self.Mannual_Control)
        self.Button_LeftArrow.setArrowType(QtCore.Qt.LeftArrow)
        self.Button_LeftArrow.setObjectName("Button_LeftArrow")
        self.horizontalLayout_2.addWidget(self.Button_LeftArrow)
        self.Button_Left = QtWidgets.QPushButton(self.Mannual_Control)
        self.Button_Left.setMaximumSize(QtCore.QSize(75, 23))
        self.Button_Left.setObjectName("Button_Left")
        self.horizontalLayout_2.addWidget(self.Button_Left)
        self.BUtton_Ahead = QtWidgets.QPushButton(self.Mannual_Control)
        self.BUtton_Ahead.setMaximumSize(QtCore.QSize(74, 23))
        self.BUtton_Ahead.setObjectName("BUtton_Ahead")
        self.horizontalLayout_2.addWidget(self.BUtton_Ahead)
        self.Button_Right = QtWidgets.QPushButton(self.Mannual_Control)
        self.Button_Right.setMaximumSize(QtCore.QSize(74, 23))
        self.Button_Right.setObjectName("Button_Right")
        self.horizontalLayout_2.addWidget(self.Button_Right)
        self.Button_RightArrow = QtWidgets.QToolButton(self.Mannual_Control)
        self.Button_RightArrow.setArrowType(QtCore.Qt.RightArrow)
        self.Button_RightArrow.setObjectName("Button_RightArrow")
        self.horizontalLayout_2.addWidget(self.Button_RightArrow)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.MineField.setTitle(_translate("Form", "Minefied(view from top)"))
        self.Automatic_Control.setTitle(_translate("Form", "Automatic Control"))
        self.Button_Reset.setText(_translate("Form", "Reset"))
        self.Button_Step.setText(_translate("Form", "Step"))
        self.Button_Auto.setText(_translate("Form", "Auto"))
        self.Experiment_Settings.setTitle(_translate("Form", "Experiment Settings"))
        self.E4.setText(_translate("Form", "Moving:"))
        self.E1.setText(_translate("Form", "Agent:"))
        self.E3.setText(_translate("Form", "Steps:"))
        self.E2.setText(_translate("Form", "Trials:"))
        self.View.setTitle(_translate("Form", "View"))
        self.TimeDelay.setText(_translate("Form", "TimeDelay"))
        self.V2.setText(_translate("Form", "Traced:"))
        self.Learnin_Parameters.setTitle(_translate("Form", "Learning Parameters"))
        self.L2.setText(_translate("Form", "Bounded TD Rule:"))
        self.L1.setText(_translate("Form", "Immediate Reward:"))
        self.Button_Patrol.setText(_translate("Form", "Patrol"))
        self.Sonar_Signal_Input.setTitle(_translate("Form", "Sonar Signal Input(Agent 0)"))
        self.AV_Sonar_Signal.setTitle(_translate("Form", "AV Sonar Signal (Agent 0)"))
        self.T1.setText(_translate("Form", "Current Bearing of Agent 0"))
        self.T2.setText(_translate("Form", "Target Bearing of Agent 0"))
        self.Mannual_Control.setTitle(_translate("Form", "Mannual Control"))
        self.Button_LeftArrow.setText(_translate("Form", "..."))
        self.Button_Left.setText(_translate("Form", "Left"))
        self.BUtton_Ahead.setText(_translate("Form", "Ahead"))
        self.Button_Right.setText(_translate("Form", "Right"))
        self.Button_RightArrow.setText(_translate("Form", "..."))
o_path = os.getcwd()
sys.path.append(o_path + '/images')
import res_rc
