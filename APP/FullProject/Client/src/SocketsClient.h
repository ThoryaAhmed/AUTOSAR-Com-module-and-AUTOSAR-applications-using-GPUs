#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_SocketsClient.h"

#include "./Platform_Types.h"

#include <QTcpsocket>
#include <QTimer>
#include <QPixmap>
#include <QImage>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\core\core.hpp>
using namespace cv;

// uint8* h_image;
// uint8* h_image_out;

class SocketsClient : public QMainWindow
{
    Q_OBJECT

public:
    SocketsClient(QWidget *parent = nullptr);
    ~SocketsClient();
    QImage imdisplay;
    // QTimer* Timer;
    int width, height, channels;
    uint8_t* gray_out; uint8_t* edges_out; uint8_t* red_roads_out;
    size_t rgb_size; size_t gray_size; size_t edges_size; size_t red_roads_size;
    uint8_t* d_rgb_in; uint8_t* d_gray_out; uint8_t* d_edges_out; uint8_t* d_red_roads_out;
    QByteArray imageData;

private slots:
    void on_StartButton_clicked();
    void on_StopButton_clicked();
    void readyRead();
    // void DisplayImage();

private:
    Ui::SocketsClientClass ui;
    QTcpSocket* serverSocket;
    
    VideoCapture cap; // OpenCV
};
