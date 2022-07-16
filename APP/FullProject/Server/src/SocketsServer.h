#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_SocketsServer.h"

#include <QTcpServer>
#include <QTcpsocket>
#include <QTimer>
#include <QString>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\core\core.hpp>
using namespace cv;

class SocketsServer : public QMainWindow
{
    Q_OBJECT

public:
    SocketsServer(QWidget *parent = nullptr);
    ~SocketsServer();
    QImage imdisplay;
    QTimer* Timer;

    int width, height, channels;
    uint8_t* gray_out; uint8_t* edges_out; uint8_t* red_roads_out;
    size_t rgb_size; size_t gray_size; size_t edges_size; size_t red_roads_size;
    uint8_t* d_rgb_in; uint8_t* d_gray_out; uint8_t* d_edges_out; uint8_t* d_red_roads_out;

private slots:
    void on_StartButton_clicked();
    void on_StopButton_clicked();
    void DisplayImage();

public slots:
    void newConnection();

private:
    Ui::SocketsServerClass ui;
    QTcpServer* server;
    QTcpSocket* socket;
    VideoCapture cap; // OpenCV
    QString word;
};
