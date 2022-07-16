#include "SocketsServer.h"
#include "kernel.cuh"

#include "Com.cuh"
#define FACTOR 0.5

uint8* h_image_out;
uint8_t* rgb_in;

SocketsServer::SocketsServer(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    cap.open("project_video.mp4");

    Timer = new QTimer(this);
    connect(Timer, SIGNAL(timeout()), this, SLOT(DisplayImage()));

    server = new QTcpServer(this);
    connect(server, SIGNAL(newConnection()), this, SLOT(newConnection()));

    word = "text...";
    server->listen(QHostAddress::Any, 1234);

    width = 1280; height = 720; channels = 3;

    rgb_size = width * height * channels * sizeof(uint8_t);

    h_image_out = (uint8_t*)malloc(rgb_size);

    cudaMalloc((void**)&d_rgb_in, rgb_size);

}

void SocketsServer::newConnection() {
    socket = server->nextPendingConnection();

    socket->write("Welcome!! from server");
    socket->flush();
    socket->waitForBytesWritten(-1);
    ui.display->setText("New Connection Is Set");
    // socket->close();

}

// try to cpy the image to another cpu buffer h_image_out
void SocketsServer::DisplayImage() {
    Mat frame;
    if (socket->isWritable()) {
        cap >> frame;
        cvtColor(frame, frame, COLOR_BGR2RGB);

        rgb_in = frame.data;
        cudaMemcpy(d_rgb_in, rgb_in, rgb_size, cudaMemcpyHostToDevice);
        QCom_SendSignal_GPU(0, d_rgb_in);

        frame.data = h_image_out;

        cv::resize(frame, frame, cv::Size(), FACTOR, FACTOR);
        unsigned int frameSize = frame.total() * frame.elemSize();

        socket->write((const char*)frame.data, frameSize);
        socket->flush();
        socket->waitForBytesWritten(-1); // returns only when written

        QImage imdisplay((uchar*)frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
        ui.display->setPixmap(QPixmap::fromImage(imdisplay));
    }
}


void SocketsServer::on_StartButton_clicked() {
    ui.display->setText("opened connection");
    Timer->start();
}

void SocketsServer::on_StopButton_clicked() {
    if (server->isListening()) {
        socket->close();
        ui.display->setText("DisConnected");
    }
}

SocketsServer::~SocketsServer()
{
    cudaFree(d_rgb_in);
}
