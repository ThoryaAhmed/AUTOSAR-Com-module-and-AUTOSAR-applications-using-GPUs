#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MainWindow.h"
#include <QTcpSocket>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:

    void on_larrow_c_stateChanged(int arg1);

    void on_rarrow_c_stateChanged(int arg1);

    void on_fuel_c_stateChanged(int arg1);

    void on_opendoor_c_stateChanged(int arg1);

    void on_light_c_stateChanged(int arg1);

    void on_battery_c_stateChanged(int arg1);

    void on_fix_c_stateChanged(int arg1);

    void on_horizontalSlider_sliderMoved(int position);

    void on_horizontalSlider_2_sliderMoved(int position);

    void on_horizontalSlider_valueChanged(int value);

    void on_horizontalSlider_2_valueChanged(int value);

public slots:
    void onReadyRead();

public:
    Ui::MainWindowClass ui;
    QTcpSocket  _socket;

};
