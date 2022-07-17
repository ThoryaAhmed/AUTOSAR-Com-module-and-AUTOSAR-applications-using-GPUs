#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MainWindow.h"
#include <QTcpServer>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void onNewConnection();
    void onSocketStateChanged(QAbstractSocket::SocketState socketState);
    void onReadyRead();

private slots:
    void on_larrow_c_stateChanged(int arg1);

    void on_rarrow_c_stateChanged(int arg1);

    void on_fuel_c_stateChanged(int arg1);

    void on_opendoor_c_stateChanged(int arg1);

    void on_light_c_stateChanged(int arg1);

    void on_battery_c_stateChanged(int arg1);

    void on_fix_c_stateChanged(int arg1);

    void on_horizontalSlider_valueChanged(int value);

    void on_horizontalSlider_2_valueChanged(int value);
public:
    Ui::MainWindowClass ui;
    QTcpServer  _server;
    QList<QTcpSocket*>  _sockets;
};
