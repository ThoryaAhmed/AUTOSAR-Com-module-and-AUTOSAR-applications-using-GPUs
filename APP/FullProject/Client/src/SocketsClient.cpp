#include "SocketsClient.h"
#include "kernel.cuh"
// #include "./gen/Com_Cfg.cuh"
#include "Com.cuh"

// uint8* h_image;
uint8* h_image_out;
uint8_t* rgb_in;

SocketsClient::SocketsClient(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    cap.open("project_video.mp4");

    // Timer = new QTimer(this);
    // connect(Timer, SIGNAL(timeout()), this, SLOT(DisplayImage()));
    // Timer->start();

    serverSocket = new QTcpSocket(this);
    connect(serverSocket, SIGNAL(readyRead()), this, SLOT(readyRead()));

    width = WIDTH * FACTOR; height = HEIGHT * FACTOR; channels = CHANNELS;

    rgb_size = width * height * channels * sizeof(uint8_t);
    gray_size = width * height * 1 * sizeof(uint8_t);
    edges_size = gray_size;
    red_roads_size = rgb_size;

    gray_out = (uint8_t*)malloc(gray_size);
    edges_out = (uint8_t*)malloc(edges_size);
    red_roads_out = (uint8_t*)malloc(rgb_size);

    cudaMalloc((void**)&d_rgb_in, rgb_size);
    cudaMalloc((void**)&d_gray_out, gray_size);
    cudaMalloc((void**)&d_edges_out, edges_size);
    cudaMalloc((void**)&d_red_roads_out, red_roads_size);

}

void SocketsClient::on_StartButton_clicked() {
    // timer should work from here
    // and at every tick a frame is sent
    serverSocket->connectToHost("127.0.0.1", 1234);
    // Timer->start();
}
void SocketsClient::on_StopButton_clicked() {
    if (serverSocket->isOpen()) {
        serverSocket->close();
    }
}


void SocketsClient::readyRead() {
    // router accumulates data
    imageData.append(serverSocket->readAll());

    // if data is ready
    if (imageData.size() >= WIDTH * HEIGHT * CHANNELS * FACTOR * FACTOR) {
        rgb_in = (uint8_t*) imageData.data();


        // Get from COM
        QCom_ReceiveSignal_GPU(0, d_rgb_in); // need to account for size

        // APP CODE
        cudaMemcpy(d_red_roads_out, d_rgb_in, red_roads_size, cudaMemcpyDeviceToDevice);
        rgb_2_gray(d_gray_out, d_rgb_in);
        doub_threshold(d_edges_out, d_gray_out, 175, 250);
        thresh_2_lanes(d_red_roads_out, d_edges_out);

        cudaMemcpy(rgb_in, d_red_roads_out, red_roads_size, cudaMemcpyDeviceToHost);
        
        // display
        QImage image((uchar*)imageData.data(), FACTOR * WIDTH, FACTOR * HEIGHT, FACTOR * WIDTH * CHANNELS, QImage::Format_RGB888);
        ui.display->setPixmap(QPixmap::fromImage(image));
        imageData.clear();
    }
}

SocketsClient::~SocketsClient()
{
    free(gray_out);
    free(edges_out);
    free(red_roads_out);

    cudaFree(d_rgb_in);
    cudaFree(d_gray_out);
    cudaFree(d_edges_out);
    cudaFree(d_red_roads_out);
}
