#include "MainWindow.h"
#include "QHostAddress"
#include "qlabel.h"
#include "cudacode.cuh"
#include "inc/Com.h"

QTcpSocket* socket_extern;


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    _socket.connectToHost(QHostAddress("127.0.0.1"), 4242);
    //_socket.connectToHost(QHostAddress("192.168.1.1"), 4242);
    socket_extern = &_socket;

    connect(&_socket, SIGNAL(readyRead()), this, SLOT(onReadyRead()));
    ui.larrow_b->setVisible(0);
    ui.rarrow_b->setVisible(0);
    ui.fuel_b->setVisible(0);
    ui.battery_b->setVisible(0);
    ui.opendoor_b->setVisible(0);
    ui.light_b->setVisible(0);
    ui.fix_b->setVisible(0);

    ui.label_2->setText(wrapper());

}

MainWindow::~MainWindow()
{}

void MainWindow::onReadyRead()
{
    QByteArray datas = _socket.readAll();
    ui.statusbar->showMessage(datas);
    //socket_extern->write(QByteArray("ok !\n"));
}

void MainWindow::on_larrow_c_stateChanged(int arg1)
{
    ui.larrow_b->setVisible(arg1);
    //QByteArray br = 'e' + QString::number(arg1).toUtf8();
    
    uint8 data = (arg1 == 2);
    Com_SendSignal_GPU(5, &data);
    
    //_socket.write(br);
    QByteArray datas = _socket.readAll();

}


void MainWindow::on_rarrow_c_stateChanged(int arg1)
{
    ui.rarrow_b->setVisible(arg1);
    //QByteArray br = 'r' + QString::number(arg1).toUtf8();

    uint8 data = (arg1 == 2);
    Com_SendSignal_GPU(6, &data);

    //_socket.write(br);
    QByteArray datas = _socket.readAll();
}


void MainWindow::on_fuel_c_stateChanged(int arg1)
{
    ui.fuel_b->setVisible(arg1);
    //QByteArray br = 'u' + QString::number(arg1).toUtf8();

    uint8 data = (arg1 == 2);
    Com_SendSignal_GPU(0, &data);

    
    //_socket.write(br);
    QByteArray datas = _socket.readAll();
}


void MainWindow::on_opendoor_c_stateChanged(int arg1)
{
    ui.opendoor_b->setVisible(arg1);
    //QByteArray br = 'p' + QString::number(arg1).toUtf8();
    
    uint8 data = (arg1 == 2);
    Com_SendSignal_GPU(1, &data);
    
    //_socket.write(br);
    QByteArray datas = _socket.readAll();
}


void MainWindow::on_light_c_stateChanged(int arg1)
{
    ui.light_b->setVisible(arg1);
    //QByteArray br = 'l' + QString::number(arg1).toUtf8();
    
    uint8 data = (arg1 == 2);
    Com_SendSignal_GPU(2, &data);

    //_socket.write(br);
    QByteArray datas = _socket.readAll();
}


void MainWindow::on_battery_c_stateChanged(int arg1)
{
    ui.battery_b->setVisible(arg1);
    //QByteArray br = 'b' + QString::number(arg1).toUtf8();
    
    uint8 data = (arg1 == 2);
    Com_SendSignal_GPU(3, &data);
    
    //_socket.write(br);
    QByteArray datas = _socket.readAll();
}


void MainWindow::on_fix_c_stateChanged(int arg1)
{
    ui.fix_b->setVisible(arg1);
    //QByteArray br = 'f' + QString::number(arg1).toUtf8();
    
    uint8 data = (arg1 == 2);
    Com_SendSignal_GPU(4, &data);
    
    //_socket.write(br);
    QByteArray datas = _socket.readAll();
}


void MainWindow::on_horizontalSlider_sliderMoved(int position)
{

}


void MainWindow::on_horizontalSlider_2_sliderMoved(int position)
{


}


void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    QString s = QString::number(value);
    ui.speed->setText(s);
    //QByteArray br = 's' + s.toUtf8();
    //_socket.write(br);

    Com_SendSignal_GPU(7, &value);

    QByteArray datas = _socket.readAll();

}


void MainWindow::on_horizontalSlider_2_valueChanged(int value)
{
    QString s = QString::number(value);
    ui.degree->setText(s);
    //QByteArray br = 'd' + s.toUtf8();
    //_socket.write(br);

    Com_SendSignal_GPU(8, &value);

    QByteArray datas = _socket.readAll();
}

